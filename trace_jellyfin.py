#!/usr/bin/env python3
"""Sample every TCP socket that carries Jellyfin traffic over an SSH tunnel.

With `ssh -L 18096:localhost:8096` from Windows, one playback session uses
three sockets on this box:

  wan     sshd -> your client on :22      the only hop on a real NIC (eno1)
  tunnel  sshd -> 127.0.0.1:8096          loopback, the tunnel's local end
  serve   jellyfin -> 127.0.0.1:<port>    loopback, jellyfin's accepted socket

The `wan` row is the one whose cwnd/rtt/retrans respond to qdisc changes; the
two loopback rows show whether Jellyfin itself is keeping up. Each sample runs
one `ss` call and appends one CSV row per socket.

  ./trace_jellyfin.py --duration 60 --output /tmp/jelly.csv
  sudo ./trace_jellyfin.py --peer 139.184.24.32   # -p needs root for names
"""

import argparse
import csv
import os
import re
import shutil
import subprocess
import sys
import time

# `ss -i` prints the congestion-control module as a bare word among the flag
# tokens, so it can only be told apart from flags like `app_limited` by name.
_CC_NAMES = {
    "astraea", "orca", "cubic", "reno", "bbr", "bbr2", "bbr3", "dctcp",
    "vegas", "westwood", "htcp", "hybla", "illinois", "lp", "nv", "scalable",
    "veno", "yeah", "bic", "cdg", "highspeed", "pcc", "aurora",
}

# key -> (csv column, scale applied to the raw value)
_SCALAR_FIELDS = {
    "cwnd": ("cwnd", 1), "ssthresh": ("ssthresh", 1), "mss": ("mss", 1),
    "pmtu": ("pmtu", 1), "rto": ("rto_ms", 1), "ato": ("ato_ms", 1),
    "unacked": ("unacked", 1), "retrans": ("retrans", 1),
    "lost": ("lost", 1), "sacked": ("sacked", 1),
    "bytes_sent": ("bytes_sent", 1), "bytes_acked": ("bytes_acked", 1),
    "bytes_received": ("bytes_received", 1),
    "bytes_retrans": ("bytes_retrans", 1),
    "segs_out": ("segs_out", 1), "segs_in": ("segs_in", 1),
    "data_segs_out": ("data_segs_out", 1),
    "data_segs_in": ("data_segs_in", 1),
    "delivered": ("delivered", 1), "busy": ("busy_ms", 1),
    "minrtt": ("minrtt_ms", 1), "rcv_space": ("rcv_space", 1),
    "rcv_ssthresh": ("rcv_ssthresh", 1),
}

_COLUMNS = [
    "wall_time", "t_s", "role", "local", "peer", "process", "cc",
    "recv_q", "send_q", "rtt_ms", "rtt_var_ms", "minrtt_ms",
    "cwnd", "ssthresh", "unacked", "retrans", "retrans_total", "lost",
    "sacked", "mss", "pmtu", "rto_ms", "ato_ms",
    "bytes_sent", "bytes_acked", "bytes_received", "bytes_retrans",
    "segs_out", "segs_in", "data_segs_out", "data_segs_in", "delivered",
    "send_bps", "pacing_bps", "delivery_bps",
    "goodput_bps", "retrans_delta", "app_limited", "busy_ms",
    "rcv_space", "rcv_ssthresh",
]


def build_filter(jellyfin_port, ssh_port, peer):
    clauses = [
        f"sport = :{jellyfin_port}", f"dport = :{jellyfin_port}",
    ]
    ssh = f"sport = :{ssh_port}"
    if peer:
        ssh = f"( {ssh} and dst {peer} )"
    clauses.append(ssh)
    return "state established ( " + " or ".join(clauses) + " )"


def classify(local_port, peer_port, jellyfin_port, ssh_port):
    if local_port == ssh_port:
        return "wan"
    if local_port == jellyfin_port:
        return "serve"
    if peer_port == jellyfin_port:
        return "tunnel"
    return "other"


def port_of(endpoint):
    return endpoint.rsplit(":", 1)[-1]


def parse_line(line):
    """One `ss -tinO` socket line -> dict of raw values."""
    fields = line.split()
    if len(fields) < 4:
        return None
    row = {
        "recv_q": fields[0], "send_q": fields[1],
        "local": fields[2], "peer": fields[3],
    }
    rest = fields[4:]

    process = ""
    tokens = []
    for token in rest:
        if token.startswith("users:(") or token.startswith("timer:("):
            process = token if token.startswith("users:(") else process
            continue
        tokens.append(token)

    if process:
        names = re.findall(r'"([^"]+)",pid=(\d+)', process)
        process = ",".join(f"{name}/{pid}" for name, pid in names)
    row["process"] = process
    row["app_limited"] = 0

    index = 0
    while index < len(tokens):
        token = tokens[index]
        # `send 92857715bps`, `pacing_rate 96192384bps`, `delivery_rate ...`
        if token in ("send", "pacing_rate", "delivery_rate") and index + 1 < len(tokens):
            key = {"send": "send_bps", "pacing_rate": "pacing_bps",
                   "delivery_rate": "delivery_bps"}[token]
            row[key] = re.sub(r"[^0-9.]", "", tokens[index + 1])
            index += 2
            continue
        if token == "app_limited":
            row["app_limited"] = 1
            index += 1
            continue
        if token in _CC_NAMES:
            row["cc"] = token
            index += 1
            continue
        if ":" in token:
            key, _, value = token.partition(":")
            if key == "rtt":
                srtt, _, var = value.partition("/")
                row["rtt_ms"], row["rtt_var_ms"] = srtt, var
            elif key == "retrans":
                # printed as retrans:<in flight>/<cumulative>
                now, _, total = value.partition("/")
                row["retrans"], row["retrans_total"] = now, total
            elif key in _SCALAR_FIELDS:
                column, _scale = _SCALAR_FIELDS[key]
                row[column] = re.sub(r"[^0-9.]", "", value)
        index += 1
    return row


def sample(ss_binary, filter_expr, with_process):
    command = [ss_binary, "-tinHO"]
    if with_process:
        command.append("-p")
    command.append(filter_expr)
    result = subprocess.run(
        command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "ss failed")
    return [line for line in result.stdout.splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", default="jellyfin_sockets.csv")
    parser.add_argument("--interval", type=float, default=0.5,
                        help="seconds between samples (default 0.5)")
    parser.add_argument("--duration", type=float, default=0.0,
                        help="seconds to trace; 0 runs until Ctrl-C")
    parser.add_argument("--jellyfin-port", type=int, default=8096)
    parser.add_argument("--ssh-port", type=int, default=22)
    parser.add_argument("--peer",
                        help="only trace the SSH session from this client IP")
    parser.add_argument("--quiet", action="store_true",
                        help="do not print the live one-line-per-sample view")
    args = parser.parse_args()

    ss_binary = shutil.which("ss")
    if not ss_binary:
        sys.exit("error: `ss` not found (install iproute2)")
    with_process = os.geteuid() == 0
    if not with_process and not args.quiet:
        print("note: not root, so the process column stays empty "
              "(re-run under sudo for pid/name)", file=sys.stderr)

    filter_expr = build_filter(
        args.jellyfin_port, args.ssh_port, args.peer)
    jelly, ssh = str(args.jellyfin_port), str(args.ssh_port)

    previous = {}
    started = time.monotonic()
    deadline = started + args.duration if args.duration > 0 else None
    samples = 0

    with open(args.output, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_COLUMNS,
                                extrasaction="ignore")
        writer.writeheader()
        try:
            while deadline is None or time.monotonic() < deadline:
                now = time.monotonic()
                for line in sample(ss_binary, filter_expr, with_process):
                    row = parse_line(line)
                    if row is None:
                        continue
                    row["role"] = classify(
                        port_of(row["local"]), port_of(row["peer"]),
                        jelly, ssh)
                    row["wall_time"] = f"{time.time():.3f}"
                    row["t_s"] = f"{now - started:.3f}"

                    # Per-socket deltas: `send`/`delivery_rate` are kernel
                    # estimates, this is what actually left the socket.
                    key = (row["local"], row["peer"])
                    acked = float(row.get("bytes_acked") or 0)
                    retrans_total = float(row.get("retrans_total") or 0)
                    last = previous.get(key)
                    if last:
                        dt = now - last[0]
                        if dt > 0:
                            row["goodput_bps"] = f"{(acked - last[1]) * 8 / dt:.0f}"
                        row["retrans_delta"] = int(retrans_total - last[2])
                    previous[key] = (now, acked, retrans_total)
                    writer.writerow(row)

                    if not args.quiet:
                        print(
                            f"{row['t_s']:>8}s {row['role']:<6} "
                            f"{row['local']:>22} -> {row['peer']:<22} "
                            f"cc={row.get('cc', ''):<8} "
                            f"cwnd={row.get('cwnd', ''):>6} "
                            f"rtt={row.get('rtt_ms', ''):>8}ms "
                            f"goodput={row.get('goodput_bps', ''):>12}bps "
                            f"rtx+{row.get('retrans_delta', 0)}",
                            flush=True)
                samples += 1
                handle.flush()
                time.sleep(max(0.0, args.interval - (time.monotonic() - now)))
        except KeyboardInterrupt:
            pass

    print(f"\nwrote {args.output} ({samples} samples, "
          f"{len(previous)} distinct sockets)", file=sys.stderr)


if __name__ == "__main__":
    main()

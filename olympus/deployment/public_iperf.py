#!/usr/bin/env python3
"""Run parallel TCP flows with a selected CC and plot both endpoints."""

import argparse
import csv
import json
import os
from pathlib import Path
import socket
import subprocess
import shutil
import time


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--server", required=True,
                   help="public iperf3 hostname (public servers change often)")
    p.add_argument("--port", type=int, default=5201)
    p.add_argument(
        "--flows", "-P", type=int, default=4,
        help="number of parallel streams in one iperf3 client session")
    p.add_argument(
        "--duration", "-t", type=int, default=40,
        help="iperf3 measurement duration in seconds (default: 40)")
    p.add_argument(
        "--cc", "--congestion", dest="cc", default="astraea",
        help="TCP congestion-control algorithm passed to iperf3 (default: astraea)")
    p.add_argument("--output", default="olympus/deployment/results/public_iperf")
    p.add_argument("--keep-history", action="store_true",
                   help="create a timestamped child instead of replacing old results")
    p.add_argument("--trace-dir", default="/tmp/olympus-deployment-traces",
                   help="trace directory used by the separately running service")
    p.add_argument(
        "--rtt-plot-max", type=float, default=120.0,
        help="upper limit of the network RTT panel in milliseconds (default: 120)")
    p.add_argument("--omit", type=int, default=2)
    p.add_argument("--ping-count", type=int, default=5)
    return p.parse_args()


def _interval_series(payload):
    result = {}
    for interval in payload.get("intervals", []):
        for stream in interval.get("streams", []):
            sid = str(stream.get("socket", stream.get("id", "flow")))
            result.setdefault(sid, [[], []])
            result[sid][0].append(float(stream.get("end", 0)))
            result[sid][1].append(float(stream.get("bits_per_second", 0)) / 1e6)
    return result


def _server_payload(client):
    value = client.get("server_output_json")
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            pass
    return {}


def _trace_score(path, max_time):
    with path.open() as handle:
        rows = [r for r in csv.DictReader(handle)
                if float(r["time_s"]) <= max_time]
    # Data sockets have sustained throughput. Control sockets and stale
    # half-closed sockets are therefore naturally excluded.
    score = sum(max(0.0, float(r["avg_thr_bps"])) for r in rows)
    return score, rows


def _plot(client, trace_paths, output, flows, max_time,
          experiment_start_wall=None, rtt_plot_max_ms=120.0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    server = _server_payload(client)
    fig, axes = plt.subplots(5, 1, figsize=(13, 16), sharex=False)
    for label, payload, style in (("client", client, "-"), ("server", server, "--")):
        for sid, (xs, ys) in _interval_series(payload).items():
            axes[0].plot(xs, ys, style, alpha=.8, label=f"{label} socket {sid}")
    axes[0].set_ylabel("Throughput (Mbit/s)")
    axes[0].set_title("iperf3 per-stream throughput: client and server JSON")
    if not server:
        axes[0].text(
            0.99, 0.04, "server JSON unavailable from this public endpoint",
            transform=axes[0].transAxes, ha="right", va="bottom",
            fontsize=8, color="dimgray")
    axes[0].legend(fontsize=7, ncol=2)
    axes[0].grid(alpha=.25)

    ranked = []
    for path in trace_paths:
        score, rows = _trace_score(path, max_time)
        if rows:
            ranked.append((score, path, rows))
    ranked.sort(key=lambda item: item[0], reverse=True)
    selected = ranked[:flows]
    for _, path, rows in selected:
        if not rows:
            continue
        name = path.stem.split("_pid_")[0]
        if (experiment_start_wall is not None
                and rows[0].get("wall_time_s") not in (None, "")):
            t = [float(r["wall_time_s"]) - experiment_start_wall
                 for r in rows]
        else:
            t = [float(r["time_s"]) for r in rows]
        axes[1].plot(t, [float(r["avg_thr_bps"]) * 8 / 1e6 for r in rows], label=name)
        axes[2].plot(t, [float(r["cwnd_after"]) for r in rows], label=name)
        axes[3].plot(
            t, [float(r["avg_urtt_us"]) / 1e3 for r in rows],
            label=name)
        axes[4].plot(t, [float(r["inference_ms"]) for r in rows], label=name)
    axes[1].set_ylabel("Kernel avg_thr (Mbit/s)")
    axes[2].set_ylabel("CWND (packets)")
    axes[3].set_ylabel("Network avg RTT (ms)")
    if rtt_plot_max_ms is not None:
        axes[3].set_ylim(0.0, float(rtt_plot_max_ms))
    axes[4].set_ylabel("Inference latency (ms)")
    axes[4].set_xlabel("Flow-worker time (s)")
    for ax in axes[1:]:
        ax.grid(alpha=.25)
        ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return [str(item[1]) for item in selected]


def main():
    args = _parse_args()
    if args.flows < 1:
        raise SystemExit("--flows must be positive")
    if args.duration <= 0:
        raise SystemExit("--duration must be positive")
    if args.omit < 0:
        raise SystemExit("--omit must be non-negative")
    if args.rtt_plot_max <= 0:
        raise SystemExit("--rtt-plot-max must be positive")
    args.cc = args.cc.strip()
    if not args.cc:
        raise SystemExit("--cc must not be empty")
    placeholders = {"your_iperf3_server", "server", "hostname", "example.com"}
    if args.server.strip().lower() in placeholders:
        raise SystemExit(
            f"--server {args.server!r} is a placeholder; provide a real "
            "iperf3 hostname, for example --server speedtest.ip-projects.de")
    output_base = Path(args.output).resolve()
    if args.keep_history:
        out = output_base / ("run_" + time.strftime("%Y%m%d-%H%M%S"))
    else:
        out = output_base
        # --output is wholly owned by this benchmark. Single-run mode replaces
        # it so stale flow CSVs/plots can never leak into the next result.
        if out.exists():
            try:
                shutil.rmtree(out)
            except PermissionError as exc:
                raise SystemExit(
                    f"cannot replace {out}: {exc}. It was likely created by "
                    "an earlier sudo run. Restore ownership with: "
                    f"sudo chown -R $USER:$(id -gn) {out.parent}") from None
    try:
        out.mkdir(parents=True, exist_ok=True)
    except PermissionError as exc:
        raise SystemExit(
            f"cannot create {out}: {exc}. This output directory was likely "
            "created by an earlier sudo run. Restore ownership with: "
            f"sudo chown -R $USER:$(id -gn) {output_base.parent}") from None
    trace_dir = Path(args.trace_dir).resolve()
    trace_dir.mkdir(parents=True, exist_ok=True)
    traces_before = set(trace_dir.glob("flow_*.csv"))

    ping_path = out / "ping.txt"
    with ping_path.open("w") as handle:
        ping = subprocess.run(
            ["ping", "-c", str(args.ping_count), args.server],
            stdout=handle, stderr=subprocess.STDOUT, text=True, check=False)
    if ping.returncode != 0:
        print(f"[public-iperf] warning: ICMP ping failed; trying TCP anyway ({ping_path})")

    try:
        addresses = socket.getaddrinfo(
            args.server, args.port, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise SystemExit(f"cannot resolve --server {args.server!r}: {exc}")
    last_error = None
    for family, socktype, proto, _, sockaddr in addresses:
        probe = socket.socket(family, socktype, proto)
        probe.settimeout(5.0)
        try:
            probe.connect(sockaddr)
            last_error = None
            break
        except OSError as exc:
            last_error = exc
        finally:
            probe.close()
    if last_error is not None:
        raise SystemExit(
            f"cannot connect to {args.server}:{args.port}: {last_error}; "
            "choose another public iperf3 server/port")

    result_path = out / "iperf3.json"
    base_command = [
        "iperf3", "--client", args.server, "--port", str(args.port),
        "--time", str(args.duration), "--omit", str(args.omit),
        "--congestion", args.cc, "--parallel", str(args.flows),
        "--json", "--get-server-output",
        "--connect-timeout", "5000",
    ]
    experiment_start_wall = time.time()
    proc = subprocess.run(
        base_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, check=False)
    result_path.write_text(proc.stdout)
    (out / "iperf3.stderr.txt").write_text(proc.stderr)
    if proc.returncode != 0:
        raise RuntimeError(
            f"iperf3 failed with {proc.returncode}: {proc.stderr.strip()}")
    client = json.loads(proc.stdout)
    server = _server_payload(client)
    (out / "iperf3_server.json").write_text(json.dumps(server, indent=2))
    # Allow the independently running workers to observe idle, exit, and flush.
    time.sleep(2.0)
    trace_paths = sorted(set(trace_dir.glob("flow_*.csv")) - traces_before)
    selected = _plot(
        client, trace_paths, out / "flows.png", args.flows,
        args.duration + args.omit + 5.0,
        experiment_start_wall=experiment_start_wall,
        rtt_plot_max_ms=args.rtt_plot_max)
    (out / "selected_flows.json").write_text(json.dumps(selected, indent=2))
    if not selected:
        print("[public-iperf] warning: no new Olympus flow traces found; "
              f"is the inference service running with --trace-dir {trace_dir}?")
    print(f"[public-iperf] congestion control: {args.cc}")
    print(f"[public-iperf] results: {out}")
    print(f"[public-iperf] plot:   {out / 'flows.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

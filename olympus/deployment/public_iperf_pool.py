#!/usr/bin/env python3
"""Run one staggered single-flow iperf3 test per server in a server pool."""

import argparse
import copy
import json
from pathlib import Path
import shutil
import socket
import subprocess
import time

from olympus.deployment.public_iperf import _plot, _server_payload


DEFAULT_SERVERS = (
    "speedtest.ip-projects.de",
    "iperf3.phoenixremoteaccess.uk",
    "iperf.astra.in.ua",
)


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--servers", nargs="+", default=list(DEFAULT_SERVERS),
        help="one independent iperf3 client is launched per server")
    parser.add_argument("--port", type=int, default=5201)
    parser.add_argument(
        "--duration", "-t", type=float, default=40.0,
        help="common wall-clock experiment horizon (default: 40)")
    parser.add_argument(
        "--flow-start-delay", "--flow-delay", type=float, default=4.0,
        help="seconds between server-flow starts (default: 4)")
    parser.add_argument(
        "--cc", "--congestion", dest="cc", default="astraea")
    parser.add_argument("--omit", type=float, default=2.0)
    parser.add_argument("--ping-count", type=int, default=3)
    parser.add_argument(
        "--output", default="results/public_iperf_pool")
    parser.add_argument("--keep-history", action="store_true")
    parser.add_argument(
        "--trace-dir", default="/tmp/olympus-deployment-traces")
    parser.add_argument(
        "--rtt-plot-max", type=float, default=120.0,
        help="upper limit of the network RTT panel in milliseconds (default: 120)")
    return parser.parse_args()


def _flow_schedule(count, duration, delay, omit):
    schedule = []
    for index in range(int(count)):
        start = index * float(delay)
        measured = float(duration) - start - float(omit)
        if measured <= 0.0:
            raise ValueError(
                f"server flow {index + 1} starts at {start:g}s, leaving no "
                f"measurement time before the {duration:g}s horizon")
        schedule.append((start, measured))
    return schedule


def _merge_intervals(entries, server_side=False):
    """Merge independently timed iperf payloads for the shared top plot."""
    intervals = []
    for entry in entries:
        payload = entry["payload"]
        if server_side:
            payload = _server_payload(payload)
        if not payload:
            continue
        offset = float(entry["data_offset_s"])
        label = f'{entry["index"]}:{entry["server"]}'
        for interval in payload.get("intervals", []):
            merged = {"streams": []}
            for stream in interval.get("streams", []):
                value = copy.deepcopy(stream)
                value["socket"] = label
                for key in ("start", "end"):
                    if key in value:
                        value[key] = float(value[key]) + offset
                merged["streams"].append(value)
            if merged["streams"]:
                intervals.append(merged)
    return {"intervals": intervals}


def _safe_name(value):
    return "".join(
        character if character.isalnum() else "_" for character in value)


def _prepare_output(args):
    base = Path(args.output).resolve()
    if args.keep_history:
        output = base / ("run_" + time.strftime("%Y%m%d-%H%M%S"))
    else:
        output = base
        if output.exists():
            try:
                shutil.rmtree(output)
            except PermissionError as exc:
                raise SystemExit(
                    f"cannot replace {output}: {exc}; restore its ownership")
    try:
        output.mkdir(parents=True, exist_ok=True)
    except PermissionError as exc:
        raise SystemExit(f"cannot create {output}: {exc}") from None
    return output


def _preflight(server, port):
    try:
        addresses = socket.getaddrinfo(
            server, port, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise RuntimeError(f"cannot resolve {server!r}: {exc}") from None
    last_error = None
    for family, socktype, proto, _, sockaddr in addresses:
        probe = socket.socket(family, socktype, proto)
        probe.settimeout(5.0)
        try:
            probe.connect(sockaddr)
            return
        except OSError as exc:
            last_error = exc
        finally:
            probe.close()
    raise RuntimeError(f"cannot connect to {server}:{port}: {last_error}")


def main():
    args = _parse_args()
    if not args.servers:
        raise SystemExit("at least one --servers value is required")
    if args.duration <= 0 or args.flow_start_delay < 0 or args.omit < 0:
        raise SystemExit(
            "duration must be positive; omit and delay must be non-negative")
    if args.rtt_plot_max <= 0:
        raise SystemExit("--rtt-plot-max must be positive")
    args.cc = args.cc.strip()
    if not args.cc:
        raise SystemExit("--cc must not be empty")
    try:
        schedule = _flow_schedule(
            len(args.servers), args.duration,
            args.flow_start_delay, args.omit)
    except ValueError as exc:
        raise SystemExit(str(exc)) from None

    output = _prepare_output(args)
    trace_dir = Path(args.trace_dir).resolve()
    trace_dir.mkdir(parents=True, exist_ok=True)
    traces_before = set(trace_dir.glob("flow_*.csv"))

    for index, server in enumerate(args.servers, start=1):
        ping_path = output / f"ping_{index}_{_safe_name(server)}.txt"
        with ping_path.open("w") as handle:
            ping = subprocess.run(
                ["ping", "-c", str(args.ping_count), server],
                stdout=handle, stderr=subprocess.STDOUT,
                text=True, check=False)
        if ping.returncode != 0:
            print(f"[public-iperf-pool] warning: ping failed for {server}")
        try:
            _preflight(server, args.port)
        except RuntimeError as exc:
            raise SystemExit(str(exc)) from None

    experiment_start_wall = time.time()
    jobs = []
    for index, (server, (target_start, measured)) in enumerate(
            zip(args.servers, schedule), start=1):
        wait_s = experiment_start_wall + target_start - time.time()
        if wait_s > 0.0:
            time.sleep(wait_s)
        launch_offset = time.time() - experiment_start_wall
        command = [
            "iperf3", "--client", server,
            "--port", str(args.port),
            "--parallel", "1",
            "--time", f"{measured:g}",
            "--omit", f"{args.omit:g}",
            "--congestion", args.cc,
            "--json", "--get-server-output",
            "--connect-timeout", "5000",
        ]
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True)
        jobs.append({
            "index": index,
            "server": server,
            "launch_offset_s": launch_offset,
            "data_offset_s": launch_offset + args.omit,
            "measured_duration_s": measured,
            "process": process,
        })
        print(
            f"[public-iperf-pool] flow {index}: server={server} "
            f"start={launch_offset:.3f}s duration={measured:g}s",
            flush=True)

    completed = []
    failures = []
    for job in jobs:
        stdout, stderr = job["process"].communicate()
        index = job["index"]
        server = job["server"]
        stem = f"flow_{index}_{_safe_name(server)}"
        (output / f"{stem}.json").write_text(stdout)
        (output / f"{stem}.stderr.txt").write_text(stderr)
        if job["process"].returncode != 0:
            failures.append(
                f"{server} exit={job['process'].returncode}: {stderr.strip()}")
            continue
        try:
            payload = json.loads(stdout)
        except json.JSONDecodeError as exc:
            failures.append(f"{server} returned invalid JSON: {exc}")
            continue
        value = {key: item for key, item in job.items() if key != "process"}
        value["payload"] = payload
        completed.append(value)

    combined_server = _merge_intervals(completed, server_side=True)
    combined_client = _merge_intervals(completed, server_side=False)
    combined_client.update({
        "olympus_server_pool": [
            {key: value for key, value in entry.items() if key != "payload"}
            for entry in completed
        ],
        "server_output_json": combined_server,
    })
    (output / "iperf3.json").write_text(
        json.dumps(combined_client, indent=2))
    (output / "iperf3_server.json").write_text(
        json.dumps(combined_server, indent=2))
    if failures:
        (output / "failures.txt").write_text("\n".join(failures) + "\n")
        raise RuntimeError("iperf pool failure: " + " | ".join(failures))

    time.sleep(2.0)
    trace_paths = sorted(
        set(trace_dir.glob("flow_*.csv")) - traces_before)
    selected = _plot(
        combined_client, trace_paths, output / "flows.png",
        len(args.servers), args.duration + 5.0,
        experiment_start_wall=experiment_start_wall,
        rtt_plot_max_ms=args.rtt_plot_max)
    (output / "selected_flows.json").write_text(
        json.dumps(selected, indent=2))
    if not selected:
        print("[public-iperf-pool] warning: no new Olympus traces found")
    print(f"[public-iperf-pool] results: {output}")
    print(f"[public-iperf-pool] plot:   {output / 'flows.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

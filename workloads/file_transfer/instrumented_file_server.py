#!/usr/bin/env python3
"""Threaded HTTP file server with per-request Linux TCP_INFO metrics."""

from __future__ import annotations

import argparse
import json
import math
import socket
import struct
import threading
import time
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit


TCP_INFO = getattr(socket, "TCP_INFO", 11)
TCP_CONGESTION = getattr(socket, "TCP_CONGESTION", 13)
METRICS_PATH = "/.olympus/metrics"


def u32(info: bytes, offset: int) -> int:
    if len(info) < offset + 4:
        return 0
    return struct.unpack_from("=I", info, offset)[0]


def tcp_snapshot(connection: socket.socket) -> dict[str, int | str]:
    """Return stable fields from Linux's struct tcp_info.

    Offsets through data_segs_out have remained append-only in the Linux UAPI:
    tcpi_unacked=24, tcpi_rtt=68, tcpi_total_retrans=100, and
    tcpi_data_segs_out=156.
    """

    info = connection.getsockopt(socket.IPPROTO_TCP, TCP_INFO, 256)
    try:
        cc_raw = connection.getsockopt(socket.IPPROTO_TCP, TCP_CONGESTION, 64)
        cc = cc_raw.split(b"\0", 1)[0].decode("ascii", errors="replace")
    except OSError:
        cc = "unknown"
    return {
        "unacked": u32(info, 24),
        "rtt_us": u32(info, 68),
        "total_retrans": u32(info, 100),
        "data_segs_out": u32(info, 156),
        "tcp_congestion_control_actual": cc,
    }


class MetricsStore:
    def __init__(self, log_path: Path) -> None:
        self.log_path = log_path
        self.lock = threading.Lock()
        self.records: dict[str, dict[int, dict[str, Any]]] = {}
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._load_existing()

    def _load_existing(self) -> None:
        if not self.log_path.is_file():
            return
        try:
            with self.log_path.open(encoding="utf-8") as handle:
                for line in handle:
                    try:
                        record = json.loads(line)
                        run_id = str(record["experiment_id"])
                        flow_id = int(record["flow_id"])
                    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                        continue
                    self.records.setdefault(run_id, {})[flow_id] = record
        except OSError as exc:
            print(f"Warning: could not read metrics log: {exc}", flush=True)

    def add(self, record: dict[str, Any]) -> None:
        run_id = str(record["experiment_id"])
        flow_id = int(record["flow_id"])
        encoded = json.dumps(record, sort_keys=True, separators=(",", ":"))
        with self.lock:
            self.records.setdefault(run_id, {})[flow_id] = record
            with self.log_path.open("a", encoding="utf-8") as handle:
                handle.write(encoded + "\n")
                handle.flush()

    def for_run(self, run_id: str) -> list[dict[str, Any]]:
        with self.lock:
            records = self.records.get(run_id, {})
            return [dict(records[key]) for key in sorted(records)]


class InstrumentedFileHandler(SimpleHTTPRequestHandler):
    server_version = "OlympusFileServer/1.0"

    def __init__(
        self,
        *args: Any,
        metrics_store: MetricsStore,
        **kwargs: Any,
    ) -> None:
        self.metrics_store = metrics_store
        super().__init__(*args, **kwargs)

    def _send_metrics(self, run_id: str) -> None:
        body = json.dumps(
            {
                "schema_version": 1,
                "experiment_id": run_id,
                "records": self.metrics_store.for_run(run_id),
            },
            sort_keys=True,
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _wait_for_acks(self, timeout_seconds: float = 5.0) -> dict[str, int | str]:
        deadline = time.monotonic() + timeout_seconds
        snapshot = tcp_snapshot(self.connection)
        while snapshot["unacked"] and time.monotonic() < deadline:
            time.sleep(0.05)
            snapshot = tcp_snapshot(self.connection)
        return snapshot

    def do_GET(self) -> None:
        parsed = urlsplit(self.path)
        query = parse_qs(parsed.query)
        if parsed.path == METRICS_PATH:
            self._send_metrics(query.get("run_id", [""])[0])
            return

        run_id = query.get("olympus_run", [""])[0]
        flow_text = query.get("flow", [""])[0]
        try:
            flow_id = int(flow_text)
        except ValueError:
            flow_id = 0

        if not run_id or flow_id < 1:
            super().do_GET()
            return

        started = time.monotonic()
        before = tcp_snapshot(self.connection)
        request_error = ""
        try:
            super().do_GET()
        except (BrokenPipeError, ConnectionResetError) as exc:
            request_error = f"{type(exc).__name__}: {exc}"
        finally:
            try:
                after = self._wait_for_acks()
                retransmissions = max(
                    0,
                    int(after["total_retrans"]) - int(before["total_retrans"]),
                )
                data_segments = max(
                    0,
                    int(after["data_segs_out"]) - int(before["data_segs_out"]),
                )
                retransmission_ratio = (
                    retransmissions * 100.0 / data_segments
                    if data_segments
                    else None
                )
                record = {
                    "schema_version": 1,
                    "experiment_id": run_id,
                    "flow_id": flow_id,
                    "client_ip": self.client_address[0],
                    "client_port": self.client_address[1],
                    "file_path": parsed.path,
                    "recorded_utc": time.strftime(
                        "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
                    ),
                    "server_elapsed_seconds": round(time.monotonic() - started, 6),
                    "tcp_congestion_control_actual": after[
                        "tcp_congestion_control_actual"
                    ],
                    "retransmissions": retransmissions,
                    "data_segments_out": data_segments,
                    "retransmission_ratio_percent": (
                        round(retransmission_ratio, 6)
                        if retransmission_ratio is not None
                        and math.isfinite(retransmission_ratio)
                        else None
                    ),
                    "server_rtt_ms": round(int(after["rtt_us"]) / 1000.0, 6),
                    "request_error": request_error,
                }
                self.metrics_store.add(record)
            except OSError as exc:
                self.log_error(
                    "TCP_INFO collection failed for run=%s flow=%s: %s",
                    run_id,
                    flow_id,
                    exc,
                )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--metrics-log", type=Path, required=True)
    args = parser.parse_args()

    if not args.directory.is_dir():
        raise SystemExit(f"File root does not exist: {args.directory}")

    store = MetricsStore(args.metrics_log)
    handler = partial(
        InstrumentedFileHandler,
        directory=str(args.directory),
        metrics_store=store,
    )
    server = ThreadingHTTPServer((args.bind, args.port), handler)
    server.daemon_threads = True
    print(
        f"Serving {args.directory} at http://{args.bind}:{args.port}/ "
        f"with TCP_INFO metrics at {METRICS_PATH}",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

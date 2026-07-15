"""One socket-I/O process per flow, backed by the shared batch server."""

import math
import json
import csv
import os
import socket
import sys
import time

import numpy as np
import yaml

# The C socket listener executes this file directly. In that mode Python adds
# olympus/deployment_batch/ to sys.path rather than the repository root, so absolute
# ``olympus.*`` imports would otherwise fail before the worker can attach.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from olympus.common.action_plugins import load_action_module
from olympus.common.state_plugins import load_state_module
from olympus.common import flow_backend
from olympus.common.pace import sleep_to_grid
from olympus.deployment_batch.protocol import recv_message, send_message


def _env(*names, default=None):
    for name in names:
        value = os.environ.get(name)
        if value not in (None, ""):
            return value
    return default


def _carry_forward_drained_metrics(raw, history):
    """Replace empty DeepCC polling windows with the last valid sample.

    ``TCP_DEEPCC_INFO`` drains its interval accumulators on every read.  A
    zero ``avg_thr``/``avg_urtt`` (and zero sample count) therefore normally
    means "no ACK sample since the previous poll", not zero throughput or a
    zero RTT.  Feeding those sentinel zeros to a state plugin creates a false
    drop/rebound sequence.  Keep the original values for tracing while making
    the dictionary passed to normalisation continuous.
    """
    sampled = {
        "avg_thr": raw.get("avg_thr", 0),
        "avg_urtt": raw.get("avg_urtt", 0),
        "cnt": raw.get("cnt", raw.get("count", 0)),
        "thr_cnt": raw.get("thr_cnt", 0),
    }
    held = {"avg_thr": False, "avg_urtt": False,
            "cnt": False, "thr_cnt": False}

    for name in ("avg_thr", "avg_urtt", "thr_cnt"):
        try:
            valid = float(raw.get(name, 0) or 0) > 0.0
        except (TypeError, ValueError):
            valid = False
        if valid:
            history[name] = raw[name]
        elif name in history:
            raw[name] = history[name]
            held[name] = True

    count_keys = [name for name in ("cnt", "count") if name in raw]
    count_value = next((raw[name] for name in count_keys
                        if float(raw.get(name, 0) or 0) > 0.0), None)
    if count_value is not None:
        history["cnt"] = count_value
    elif count_keys and "cnt" in history:
        for name in count_keys:
            raw[name] = history["cnt"]
        held["cnt"] = True

    return sampled, held


def run():
    config_path = _env("OLYMPUS_CONFIG", "SAO_CONFIG", "ASTRAEA_CONFIG")
    if not config_path:
        raise RuntimeError("OLYMPUS_CONFIG is required")
    with open(config_path) as handle:
        cfg = yaml.safe_load(handle) or {}
    dep = cfg.get("deployment", {}) or {}
    worker_cfg = dep.get("workers", {}) or {}
    runtime = cfg.get("runtime", {}) or {}
    algorithm = str(runtime.get("algorithm", "td3"))
    selected = ((cfg.get("algorithms", {}) or {}).get(algorithm, {}) or {})
    agent_cfg = dict(selected.get("agent", {}) or {})
    agent_cfg.update(cfg.get("agent", {}) or {})
    flow_fd = int(_env("OLYMPUS_FLOW_FD", "OC_FLOW_FD", "ASTRAEA_FLOW_FD"))
    flow_id = _env("OLYMPUS_FLOW_ID", "OC_FLOW_ID", "ASTRAEA_FLOW_ID")
    socket_path = _env("OLYMPUS_INFERENCE_SOCKET", default=dep.get("socket_path", "/tmp/olympus-inference.sock"))
    interval_s = float(worker_cfg.get(
        "interval_ms", agent_cfg.get("interval_ms", 20))) / 1000.0
    # A missed control deadline should fail safe, but 10 ms was below observed
    # scheduler/model tail latency and caused workers to exit/re-attach. Keep
    # this comfortably above the 20 ms control cadence; it is an error bound,
    # not the normal batching delay.
    timeout_s = float(worker_cfg.get("request_timeout_ms", 250)) / 1000.0
    cwnd_min = int(worker_cfg.get(
        "cwnd_min", agent_cfg.get("cwnd_min", 10)))
    cwnd_max = int(worker_cfg.get(
        "cwnd_max", agent_cfg.get("cwnd_max", 10000)))
    action_name = str(runtime.get("action", "cwnd_multiplier"))
    action_options = (cfg.get("actions", {}) or {}).get(action_name, {}) or {}
    os.environ["SAO_ACTION_NAME"] = action_name
    os.environ["SAO_ACTION_CONFIG"] = json.dumps(action_options)
    action_plugin = load_action_module(runtime.get("action"))
    state_name = str(_env(
        "OLYMPUS_STATE_NAME",
        default=runtime.get(
            "state", "tempest" if algorithm == "ma_dreamer"
            else "default_orca")))
    state_module = load_state_module(algorithm, state_name)
    state_dim = int(state_module.STATE_DIM)

    trace_file = None
    trace_writer = None
    trace_dir = _env("OLYMPUS_TRACE_DIR", default=worker_cfg.get("trace_dir"))
    if trace_dir:
        os.makedirs(trace_dir, exist_ok=True)
        trace_path = os.path.join(trace_dir, f"flow_{flow_id}_pid_{os.getpid()}.csv")
        trace_file = open(trace_path, "w", newline="")
        trace_writer = csv.writer(trace_file)
        trace_writer.writerow([
            "time_s", "wall_time_s", "seq", "action", "cwnd_before", "cwnd_after",
            "avg_thr_bps", "avg_urtt_us", "srtt_us", "min_rtt_us",
            "loss_bytes", "packets_out", "retrans_out",
            "sampled_avg_thr_bps", "sampled_avg_urtt_us",
            "sampled_cnt", "sampled_thr_cnt",
            "held_avg_thr", "held_avg_urtt", "held_cnt", "held_thr_cnt",
            "inference_ms",
        ] + [f"s{i}" for i in range(state_dim)])

    conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    conn.settimeout(float(worker_cfg.get("connect_timeout_ms", 5000)) / 1000.0)
    conn.connect(socket_path)
    conn.settimeout(timeout_s)
    seq = 0
    prev_urtt, prev_cwnd, peak_thr = 0.0, cwnd_min, 0.0
    drained_history = {}
    next_tick = time.monotonic()
    started = next_tick
    try:
        while True:
            raw = flow_backend.get_tcp_deepcc_info(flow_fd)
            sampled, held = _carry_forward_drained_metrics(
                raw, drained_history)
            avg_thr = float(raw.get("avg_thr", 0))
            peak_thr = max(peak_thr, avg_thr)
            raw.update(prev_urtt=prev_urtt, prev_cwnd=prev_cwnd,
                       peak_thr=peak_thr, interval_ms=interval_s * 1000.0)
            # State plugins receive only ordinary scalar values.  The worker
            # owns normalisation because stateful trackers must be per-flow.
            clean = {k: (v.item() if isinstance(v, np.generic) else v)
                     for k, v in raw.items()
                     if isinstance(v, (str, bool, int, float, np.generic))}
            # Normalize here, not in the shared model process. Stateful
            # preprocessors (Tempest/Kalman) must remain process-local exactly
            # as they were during training; only the tensor inference batches.
            normalized_state = np.asarray(
                state_module.normalize_state(clean), dtype=np.float32
            ).reshape(-1)
            if normalized_state.size != state_dim:
                raise RuntimeError(
                    f"state plugin {state_name!r} returned "
                    f"{normalized_state.size} values; expected {state_dim}")
            state_payload = normalized_state.tolist()
            seq += 1
            inference_started = time.monotonic()
            send_message(conn, {"type": "infer", "flow_id": flow_id,
                                "seq": seq,
                                "state": state_payload})
            response = recv_message(conn)
            inference_ms = (time.monotonic() - inference_started) * 1000.0
            action = float("nan")
            current = int(raw.get("cwnd", prev_cwnd))
            new_cwnd = current
            if response.get("ok") and response.get("seq") == seq:
                action = float(response["action"])
                new_cwnd = action_plugin.apply_cwnd(
                    current, action, cwnd_min, cwnd_max)
                flow_backend.set_cwnd(flow_fd, int(np.clip(new_cwnd, cwnd_min, cwnd_max)))
            if trace_writer:
                trace_writer.writerow([
                    f"{time.monotonic() - started:.6f}",
                    f"{time.time():.6f}", seq, action, current,
                    int(new_cwnd), raw.get("avg_thr", 0), raw.get("avg_urtt", 0),
                    raw.get("srtt_us", 0), raw.get("min_rtt", 0),
                    raw.get("loss_bytes", 0), raw.get("packets_out", 0),
                    raw.get("retrans_out", 0),
                    sampled["avg_thr"], sampled["avg_urtt"],
                    sampled["cnt"], sampled["thr_cnt"],
                    int(held["avg_thr"]), int(held["avg_urtt"]),
                    int(held["cnt"]), int(held["thr_cnt"]),
                    f"{inference_ms:.4f}",
                ] + state_payload)
                if seq % 50 == 0:
                    trace_file.flush()
            urtt = float(raw.get("avg_urtt", 0))
            if urtt > 0:
                prev_urtt = urtt
            prev_cwnd = int(raw.get("cwnd", prev_cwnd))
            # Skip missed grid points. Sleeping against a stale target caused
            # a startup burst of many back-to-back multiplicative CWND actions.
            next_tick = sleep_to_grid(next_tick, interval_s)
    except (EOFError, OSError, flow_backend.SimulationFinished) as exc:
        print(f"[deployment-worker flow={flow_id}] exiting: "
              f"{type(exc).__name__}: {exc}", flush=True)
    finally:
        try:
            conn.settimeout(0.1)
            send_message(conn, {"type": "detach", "flow_id": flow_id})
        except OSError:
            pass
        conn.close()
        if trace_file:
            trace_file.close()


if __name__ == "__main__":
    run()

"""One complete model-owning process for one inherited TCP socket."""

import csv
import json
import os
import sys
import time

# Each flow is already an OS process. Keep native math libraries from creating
# a second thread pool inside every model-owning worker.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from olympus.common import flow_backend
from olympus.common.action_plugins import load_action_module
from olympus.common.pace import sleep_to_grid
from olympus.common.state_plugins import load_state_module
from olympus.deployment.model import FlowPolicy


def _env(*names, default=None):
    for name in names:
        value = os.environ.get(name)
        if value not in (None, ""):
            return value
    return default


def _selected_agent(cfg, algorithm):
    selected = ((cfg.get("algorithms", {}) or {}).get(algorithm, {}) or {})
    agent = dict(selected.get("agent", {}) or {})
    agent.update(cfg.get("agent", {}) or {})
    return agent


def _carry_forward_drained_metrics(raw, history):
    """Hold DeepCC drain-on-read accumulators across empty polling windows."""
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
    count_value = None
    for name in count_keys:
        try:
            if float(raw.get(name, 0) or 0) > 0.0:
                count_value = raw[name]
                break
        except (TypeError, ValueError):
            pass
    if count_value is not None:
        history["cnt"] = count_value
    elif count_keys and "cnt" in history:
        for name in count_keys:
            raw[name] = history["cnt"]
        held["cnt"] = True
    return sampled, held


def run():
    config_path = _env("OLYMPUS_CONFIG", "SAO_CONFIG", "ASTRAEA_CONFIG")
    checkpoint = _env("OLYMPUS_CHECKPOINT", "SAO_CHECKPOINT",
                      "ASTRAEA_MODEL")
    if not config_path:
        raise RuntimeError("OLYMPUS_CONFIG is required")
    if not checkpoint:
        raise RuntimeError("OLYMPUS_CHECKPOINT is required")
    with open(config_path) as handle:
        cfg = yaml.safe_load(handle) or {}

    dep = cfg.get("deployment", {}) or {}
    worker_cfg = dep.get("workers", {}) or {}
    runtime = cfg.get("runtime", {}) or {}
    algorithm = str(runtime.get("algorithm", "td3"))
    agent_cfg = _selected_agent(cfg, algorithm)
    interval_ms = float(worker_cfg.get(
        "interval_ms", agent_cfg.get("interval_ms", 20)))
    interval_s = interval_ms / 1000.0
    cwnd_min = int(worker_cfg.get(
        "cwnd_min", agent_cfg.get("cwnd_min", 10)))
    cwnd_max = int(worker_cfg.get(
        "cwnd_max", agent_cfg.get("cwnd_max", 10000)))
    flow_fd = int(_env(
        "OLYMPUS_FLOW_FD", "OC_FLOW_FD", "ASTRAEA_FLOW_FD"))
    flow_id = _env(
        "OLYMPUS_FLOW_ID", "OC_FLOW_ID", "ASTRAEA_FLOW_ID")

    # Orca hands the connection to the agent only after startup: the base CC
    # (cubic, per `listener_cc` in olympus/orca.yaml) owns cwnd for a fixed
    # warmup, then the agent takes over permanently. Deploying without this
    # puts the actor in a regime it never saw in replay.
    #
    # Unlike algorithms/orca/worker.py, the handoff here is purely time-based:
    # the training worker also releases on the first loss (its proxy for the
    # end of slow start), which on a lossy link can fire in tens of ms.
    warmup_default = algorithm == "orca"
    cubic_warmup = bool(worker_cfg.get(
        "cubic_warmup", agent_cfg.get("cubic_warmup", warmup_default)))
    cubic_warmup_s = float(worker_cfg.get(
        "cubic_warmup_s",
        worker_cfg.get("cubic_warmup_max_s",
                       agent_cfg.get("cubic_warmup_max_s", 1.0))))

    action_name = str(runtime.get("action", "cwnd_multiplier"))
    action_options = (cfg.get("actions", {}) or {}).get(
        action_name, {}) or {}
    os.environ["SAO_ACTION_NAME"] = action_name
    os.environ["SAO_ACTION_CONFIG"] = json.dumps(action_options)
    action_plugin = load_action_module(action_name)
    state_name = str(_env(
        "OLYMPUS_STATE_NAME",
        default=runtime.get(
            "state", "tempest" if algorithm == "ma_dreamer"
            else "default_orca")))
    state_module = load_state_module(algorithm, state_name)
    state_dim = int(state_module.STATE_DIM)
    policy = FlowPolicy(
        cfg, os.path.abspath(checkpoint), dep.get("device", "cpu"))

    trace_file = None
    trace_writer = None
    trace_dir = _env("OLYMPUS_TRACE_DIR", default=worker_cfg.get("trace_dir"))
    if trace_dir:
        os.makedirs(trace_dir, exist_ok=True)
        trace_path = os.path.join(
            trace_dir, f"flow_{flow_id}_pid_{os.getpid()}.csv")
        trace_file = open(trace_path, "w", newline="")
        trace_writer = csv.writer(trace_file)
        trace_writer.writerow([
            "time_s", "wall_time_s", "seq", "action", "cwnd_before", "cwnd_after",
            "avg_thr_bps", "avg_urtt_us", "srtt_us", "min_rtt_us",
            "loss_bytes", "packets_out", "retrans_out",
            "sampled_avg_thr_bps", "sampled_avg_urtt_us",
            "sampled_cnt", "sampled_thr_cnt",
            "held_avg_thr", "held_avg_urtt", "held_cnt", "held_thr_cnt",
            "warmup", "inference_ms",
        ] + [f"s{i}" for i in range(state_dim)])

    seq = 0
    prev_urtt, prev_cwnd, peak_thr = 0.0, cwnd_min, 0.0
    agent_ready = not cubic_warmup
    drained_history = {}
    next_tick = time.monotonic()
    started = next_tick
    print(f"[deployment-worker flow={flow_id} pid={os.getpid()}] "
          f"loaded {algorithm} checkpoint={os.path.basename(checkpoint)}"
          + (f"; base CC owns the first {cubic_warmup_s}s"
             if cubic_warmup else ""),
          flush=True)
    try:
        while True:
            raw = flow_backend.get_tcp_deepcc_info(flow_fd)
            sampled, held = _carry_forward_drained_metrics(
                raw, drained_history)
            avg_thr = float(raw.get("avg_thr", 0))
            peak_thr = max(peak_thr, avg_thr)
            raw.update(prev_urtt=prev_urtt, prev_cwnd=prev_cwnd,
                       peak_thr=peak_thr, interval_ms=interval_ms)
            clean = {
                key: (value.item() if isinstance(value, np.generic) else value)
                for key, value in raw.items()
                if isinstance(value, (str, bool, int, float, np.generic))
            }
            normalized_state = np.asarray(
                state_module.normalize_state(clean), dtype=np.float32
            ).reshape(-1)
            if normalized_state.size != state_dim:
                raise RuntimeError(
                    f"state plugin {state_name!r} returned "
                    f"{normalized_state.size} values; expected {state_dim}")
            state_payload = normalized_state.tolist()
            seq += 1
            current = int(raw.get("cwnd", prev_cwnd))
            elapsed = time.monotonic() - started

            if not agent_ready and elapsed >= cubic_warmup_s:
                # Fixed warmup: the base CC keeps the connection for the whole
                # window whether or not it has already lost a packet.
                agent_ready = True
                print(f"[deployment-worker flow={flow_id}] base CC -> "
                      f"agent handoff at t={elapsed:.2f}s cwnd={current} "
                      f"(warmup {cubic_warmup_s}s elapsed)", flush=True)

            if not agent_ready:
                # Hands off: no set_cwnd at all, so the kernel CC keeps driving
                # the ramp exactly as it does during training.
                action, new_cwnd, inference_ms = 0.0, current, 0.0
            else:
                inference_started = time.monotonic()
                action = policy.infer(state_payload)
                inference_ms = (
                    time.monotonic() - inference_started) * 1000.0
                new_cwnd = action_plugin.apply_cwnd(
                    current, action, cwnd_min, cwnd_max)
                flow_backend.set_cwnd(
                    flow_fd, int(np.clip(new_cwnd, cwnd_min, cwnd_max)))

            if trace_writer:
                trace_writer.writerow([
                    f"{time.monotonic() - started:.6f}",
                    f"{time.time():.6f}", seq, action,
                    current, int(new_cwnd), raw.get("avg_thr", 0),
                    raw.get("avg_urtt", 0), raw.get("srtt_us", 0),
                    raw.get("min_rtt", 0), raw.get("loss_bytes", 0),
                    raw.get("packets_out", 0), raw.get("retrans_out", 0),
                    sampled["avg_thr"], sampled["avg_urtt"],
                    sampled["cnt"], sampled["thr_cnt"],
                    int(held["avg_thr"]), int(held["avg_urtt"]),
                    int(held["cnt"]), int(held["thr_cnt"]),
                    int(not agent_ready), f"{inference_ms:.4f}",
                ] + state_payload)
                if seq % 50 == 0:
                    trace_file.flush()
            urtt = float(raw.get("avg_urtt", 0))
            if urtt > 0:
                prev_urtt = urtt
            prev_cwnd = int(raw.get("cwnd", prev_cwnd))
            next_tick = sleep_to_grid(next_tick, interval_s)
    except (OSError, flow_backend.SimulationFinished) as exc:
        print(f"[deployment-worker flow={flow_id}] exiting: "
              f"{type(exc).__name__}: {exc}", flush=True)
    finally:
        if trace_file:
            trace_file.close()


if __name__ == "__main__":
    run()

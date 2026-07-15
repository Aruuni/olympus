"""Launch socket discovery; every discovered flow owns a complete model worker."""

import argparse
import fcntl
import os
import signal
import subprocess
import sys

import yaml


def _acquire_service_lock(path):
    handle = open(path, "a+")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        raise SystemExit(
            f"another Olympus deployment service holds {path}; stop it first")
    return handle


def _checkpoint_path(cfg, override):
    runtime = cfg.get("runtime", {}) or {}
    algorithm = str(runtime.get("algorithm", "td3"))
    selected = ((cfg.get("algorithms", {}) or {}).get(algorithm, {}) or {})
    return (override
            or (cfg.get("deployment", {}) or {}).get("checkpoint")
            or (cfg.get("training", {}) or {}).get("resume_from")
            or (selected.get("training", {}) or {}).get("resume_from"))


def _checkpoint_state_name(checkpoint, fallback):
    import torch
    payload = torch.load(
        checkpoint, map_location="cpu", weights_only=False)
    return str((payload.get("state_meta") or {}).get("state_name") or fallback)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument(
        "--trace-dir", default="/tmp/olympus-deployment-traces",
        help="directory where per-socket model workers write CSV traces")
    args = parser.parse_args()

    config = os.path.abspath(args.config)
    with open(config) as handle:
        cfg = yaml.safe_load(handle) or {}
    checkpoint = _checkpoint_path(cfg, args.checkpoint)
    if not checkpoint:
        raise SystemExit("checkpoint is required")
    checkpoint = os.path.abspath(checkpoint)
    if not os.path.isfile(checkpoint):
        raise SystemExit(f"checkpoint not found: {checkpoint}")

    dep = cfg.get("deployment", {}) or {}
    service_lock = _acquire_service_lock(
        os.path.abspath(dep.get(
            "lock_path", "/tmp/olympus-deployment.lock")))
    discovery = dep.get("discovery", {}) or {}
    runtime = cfg.get("runtime", {}) or {}
    algorithm = str(runtime.get("algorithm", "td3"))
    state_name = _checkpoint_state_name(
        checkpoint, runtime.get("state", "default_orca"))
    root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    listener = os.path.abspath(discovery.get(
        "listener", os.path.join(root, "astraea_listener")))
    worker = os.path.join(root, "olympus", "deployment", "worker.py")
    inherited_pythonpath = os.environ.get("PYTHONPATH", "")
    pythonpath = (root if not inherited_pythonpath
                  else root + os.pathsep + inherited_pythonpath)
    env = dict(
        os.environ,
        ASTRAEA_PYTHON=sys.executable,
        ASTRAEA_CONFIG=config,
        ASTRAEA_MODEL=checkpoint,
        OLYMPUS_CONFIG=config,
        OLYMPUS_CHECKPOINT=checkpoint,
        OLYMPUS_STATE_NAME=state_name,
        OLYMPUS_TRACE_DIR=os.path.abspath(args.trace_dir),
        PYTHONPATH=pythonpath,
    )
    command = [
        listener,
        "--mode", str(discovery.get("mode", "mininet")),
        "--cc-name", str(discovery.get("cc_algorithm", "astraea")),
        "--script", worker,
        "--config", config,
        "--model", checkpoint,
        "--scan-ms", str(discovery.get("scan_ms", 10)),
        "--ipv4-only", "1" if discovery.get("ipv4_only", True) else "0",
    ]
    print(f"[deployment] per-socket {algorithm} workers; "
          f"checkpoint={checkpoint}", flush=True)
    child = subprocess.Popen(
        command, env=env, cwd=root, start_new_session=True)
    try:
        return child.wait()
    finally:
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            child.wait()
        service_lock.close()


if __name__ == "__main__":
    raise SystemExit(main())

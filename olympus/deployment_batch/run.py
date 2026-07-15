"""Start the dynamic-batch server and socket discovery listener."""

import argparse
import fcntl
import os
import signal
import subprocess
import sys
import time

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


def _positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return value


def _checkpoint_path(cfg, override):
    runtime = cfg.get("runtime", {}) or {}
    algorithm = str(runtime.get("algorithm", "td3"))
    selected = ((cfg.get("algorithms", {}) or {}).get(algorithm, {}) or {})
    return (override
            or (cfg.get("deployment", {}) or {}).get("checkpoint")
            or (cfg.get("training", {}) or {}).get("resume_from")
            or (selected.get("training", {}) or {}).get("resume_from"))


def _checkpoint_state_name(checkpoint, fallback):
    if not checkpoint:
        return str(fallback)
    import torch
    payload = torch.load(os.path.abspath(checkpoint), map_location="cpu",
                         weights_only=False)
    return str((payload.get("state_meta") or {}).get("state_name") or fallback)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument(
        "--max-batch-size", type=_positive_int,
        help=("override deployment.batching.max_batch_size; use 1 to "
              "disable cross-flow batching for an A/B test"))
    parser.add_argument(
        "--trace-dir", default="/tmp/olympus-deployment-traces",
        help="directory where per-socket workers write CSV traces")
    args = parser.parse_args()
    config = os.path.abspath(args.config)
    with open(config) as handle:
        cfg = yaml.safe_load(handle) or {}
    dep = cfg.get("deployment", {}) or {}
    service_lock = _acquire_service_lock(
        os.path.abspath(dep.get(
            "lock_path", "/tmp/olympus-deployment.lock")))
    discovery = dep.get("discovery", {}) or {}
    socket_path = os.path.abspath(dep.get("socket_path", "/tmp/olympus-inference.sock"))
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    listener = os.path.abspath(discovery.get("listener", os.path.join(root, "astraea_listener")))
    worker = os.path.join(root, "olympus", "deployment_batch", "worker.py")
    checkpoint = _checkpoint_path(cfg, args.checkpoint)
    runtime = cfg.get("runtime", {}) or {}
    state_name = _checkpoint_state_name(
        checkpoint, runtime.get("state", "default_orca"))
    server_cmd = [sys.executable, "-m", "olympus.deployment_batch.server", "--config", config]
    if checkpoint:
        server_cmd += ["--checkpoint", os.path.abspath(checkpoint)]
    if args.max_batch_size is not None:
        server_cmd += ["--max-batch-size", str(args.max_batch_size)]
    server = subprocess.Popen(server_cmd, cwd=root, start_new_session=True)
    deadline = time.monotonic() + 10
    while not os.path.exists(socket_path):
        if server.poll() is not None:
            raise SystemExit("inference server exited during startup")
        if time.monotonic() >= deadline:
            server.terminate()
            raise SystemExit(f"timed out waiting for {socket_path}")
        time.sleep(0.05)
    inherited_pythonpath = os.environ.get("PYTHONPATH", "")
    pythonpath = root if not inherited_pythonpath else root + os.pathsep + inherited_pythonpath
    env = dict(os.environ, ASTRAEA_PYTHON=sys.executable,
               OLYMPUS_CONFIG=config, OLYMPUS_INFERENCE_SOCKET=socket_path,
               OLYMPUS_STATE_NAME=state_name,
               OLYMPUS_TRACE_DIR=os.path.abspath(args.trace_dir),
               PYTHONPATH=pythonpath,
               # Compatibility with astraea_listener's existing child env.
               ASTRAEA_CONFIG=config)
    command = [listener, "--mode", str(discovery.get("mode", "mininet")),
               "--cc-name", str(discovery.get("cc_algorithm", "astraea")),
               "--script", worker, "--config", config, "--model", "unused",
               "--scan-ms", str(discovery.get("scan_ms", 10)),
               "--ipv4-only", "1" if discovery.get("ipv4_only", True) else "0"]
    child = subprocess.Popen(command, env=env, cwd=root, start_new_session=True)
    try:
        return child.wait()
    finally:
        for proc in (child, server):
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
        try:
            server.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(server.pid, signal.SIGKILL)
            server.wait()
        service_lock.close()


if __name__ == "__main__":
    raise SystemExit(main())

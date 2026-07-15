"""Central low-latency dynamic batch inference server."""

import argparse
import os
import queue
import socket
import threading
import time

import yaml

from olympus.deployment_batch.model import BatchPolicy
from olympus.deployment_batch.protocol import recv_message, send_message


def _positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return value


class InferenceServer:
    def __init__(self, policy, socket_path, max_batch_size=64, max_wait_us=1000):
        self.policy = policy
        self.socket_path = os.path.abspath(socket_path)
        self.max_batch_size = max(1, int(max_batch_size))
        self.max_wait_s = max(0, int(max_wait_us)) / 1_000_000.0
        self.requests = queue.Queue()
        self.stop = threading.Event()
        self.listener = None

    def _client(self, conn):
        with conn:
            while not self.stop.is_set():
                try:
                    message = recv_message(conn)
                except (EOFError, OSError):
                    return
                if message.get("type") == "detach":
                    self.policy.detach(message.get("flow_id"))
                    return
                done = queue.Queue(maxsize=1)
                self.requests.put((message, done))
                try:
                    response = done.get()
                    send_message(conn, response)
                except OSError:
                    return

    def _batch_loop(self):
        while not self.stop.is_set():
            try:
                first = self.requests.get(timeout=0.1)
            except queue.Empty:
                continue
            batch = [first]
            deadline = time.monotonic() + self.max_wait_s
            while len(batch) < self.max_batch_size:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    batch.append(self.requests.get(timeout=remaining))
                except queue.Empty:
                    break
            messages = [item[0] for item in batch]
            try:
                actions = self.policy.infer(messages)
                for (message, done), action in zip(batch, actions):
                    done.put({"ok": True, "seq": message.get("seq"), "action": action})
            except Exception as exc:
                for message, done in batch:
                    done.put({"ok": False, "seq": message.get("seq"), "error": str(exc)})

    def serve_forever(self):
        os.makedirs(os.path.dirname(self.socket_path), exist_ok=True)
        try:
            os.unlink(self.socket_path)
        except FileNotFoundError:
            pass
        self.listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.listener.bind(self.socket_path)
        os.chmod(self.socket_path, 0o660)
        self.listener.listen(256)
        threading.Thread(target=self._batch_loop, daemon=True).start()
        try:
            while not self.stop.is_set():
                conn, _ = self.listener.accept()
                threading.Thread(target=self._client, args=(conn,), daemon=True).start()
        finally:
            self.close()

    def close(self):
        self.stop.set()
        if self.listener:
            self.listener.close()
        try:
            os.unlink(self.socket_path)
        except FileNotFoundError:
            pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument(
        "--max-batch-size", type=_positive_int,
        help=("override deployment.batching.max_batch_size; use 1 to "
              "process every request independently"))
    args = parser.parse_args()
    with open(args.config) as handle:
        cfg = yaml.safe_load(handle) or {}
    deployment = cfg.get("deployment", {}) or {}
    batch = deployment.get("batching", {}) or {}
    runtime = cfg.get("runtime", {}) or {}
    algorithm = str(runtime.get("algorithm", "td3"))
    selected = ((cfg.get("algorithms", {}) or {}).get(algorithm, {}) or {})
    checkpoint = (
        args.checkpoint
        or deployment.get("checkpoint")
        or (cfg.get("training", {}) or {}).get("resume_from")
        or (selected.get("training", {}) or {}).get("resume_from"))
    if not checkpoint:
        raise SystemExit("checkpoint is required (deployment.checkpoint or training.resume_from)")
    policy = BatchPolicy(cfg, os.path.abspath(checkpoint), deployment.get("device", "cpu"))
    max_batch_size = (
        args.max_batch_size if args.max_batch_size is not None
        else batch.get("max_batch_size", 64))
    server = InferenceServer(policy, deployment.get("socket_path", "/tmp/olympus-inference.sock"),
                             max_batch_size, batch.get("max_wait_us", 1000))
    print(f"[deployment-batch] serving {policy.algorithm} on {server.socket_path} "
          f"max_batch_size={server.max_batch_size}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()

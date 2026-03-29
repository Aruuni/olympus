#!/usr/bin/env python3
import argparse
import json
import os
import signal
import socket
import sys
import threading
import time
from multiprocessing import get_context

# -----------------------------------------------------------------------------
# Simulated heavy preload
# Replace preload_heavy() and FlowService with your real Astraea bits later.
# -----------------------------------------------------------------------------

PRELOAD_INFO = {}


def preload_heavy():
    t0 = time.perf_counter()

    # Replace these with your real heavy imports / model load later.
    # Example:
    # import numpy as np
    # import tensorflow.compat.v1 as tf
    # tf.disable_v2_behavior()
    # from python.agent.agent import Agent
    # from python.agent.definitions import transform_state, STATE_DIM, ACTION_DIM, GLOBAL_DIM
    # from python.helpers.utils import Params
    #
    # and maybe build/load the model here if you want it inherited by forked children.

    time.sleep(1.0)  # demo: pretend this is TensorFlow/model startup cost

    t1 = time.perf_counter()
    PRELOAD_INFO["preload_s"] = t1 - t0


class FlowService:
    def __init__(self, service_id: int, flow_id: int, fd: int):
        self.service_id = service_id
        self.flow_id = flow_id
        self.fd = fd
        self.control = False
        self.started_at = None
        self.stop_evt = threading.Event()
        self.thread = None

    def start(self):
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        while not self.stop_evt.is_set():
            elapsed = "off" if self.started_at is None else f"{time.monotonic() - self.started_at:.3f}s"
            print(
                f"[svc] service={self.service_id} flow={self.flow_id} "
                f"fd={self.fd} control={self.control} t={elapsed}",
                flush=True,
            )
            time.sleep(0.2)

    def set_control(self, control: bool):
        control = bool(control)
        if control and not self.control:
            self.started_at = time.monotonic()
        elif not control:
            self.started_at = None
        self.control = control

    def status(self):
        return {
            "service_id": self.service_id,
            "flow_id": self.flow_id,
            "fd": self.fd,
            "control": self.control,
            "elapsed_s": None if self.started_at is None else (time.monotonic() - self.started_at),
        }

    def stop(self):
        self.stop_evt.set()
        if self.thread is not None:
            self.thread.join(timeout=0.5)
        try:
            if self.fd is not None and self.fd >= 0:
                os.close(self.fd)
        except OSError:
            pass


# -----------------------------------------------------------------------------
# Child entrypoint
# -----------------------------------------------------------------------------

def child_main(conn, service_id: int, flow_id: int, fd: int):
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    svc = FlowService(service_id=service_id, flow_id=flow_id, fd=fd)
    svc.start()
    conn.send({"type": "READY", "service_id": service_id})

    while True:
        try:
            msg = conn.recv()
        except EOFError:
            break

        cmd = msg.get("cmd")
        try:
            if cmd == "SET_CONTROL":
                svc.set_control(bool(msg["control"]))
                conn.send({"type": "OK", "control": svc.control})
            elif cmd == "STATUS":
                conn.send({"type": "OK", "status": svc.status()})
            elif cmd == "DETACH":
                svc.stop()
                conn.send({"type": "OK"})
                break
            else:
                conn.send({"type": "ERR", "error": f"unknown cmd {cmd}"})
        except Exception as e:
            conn.send({"type": "ERR", "error": str(e)})

    try:
        svc.stop()
    except Exception:
        pass


# -----------------------------------------------------------------------------
# Broker
# -----------------------------------------------------------------------------

class ChildRef:
    def __init__(self, service_id, proc, conn, flow_id, fd):
        self.service_id = service_id
        self.proc = proc
        self.conn = conn
        self.flow_id = flow_id
        self.fd = fd


class Broker:
    def __init__(self, sock_path: str):
        self.sock_path = sock_path
        self.ctx = get_context("fork")
        self.lock = threading.RLock()
        self.stop_evt = threading.Event()
        self.next_service_id = 1
        self.children = {}
        self.flow_to_service = {}

    def start(self):
        preload_heavy()
        print(f"[broker] preloaded in {PRELOAD_INFO['preload_s']:.3f}s", flush=True)

        try:
            os.unlink(self.sock_path)
        except FileNotFoundError:
            pass

        self.server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.server.bind(self.sock_path)
        self.server.listen(64)
        print(f"[broker] listening on {self.sock_path}", flush=True)

        while not self.stop_evt.is_set():
            try:
                conn, _ = self.server.accept()
            except OSError:
                break
            threading.Thread(target=self._handle_client, args=(conn,), daemon=True).start()

    def stop(self):
        self.stop_evt.set()
        try:
            self.server.close()
        except Exception:
            pass

        with self.lock:
            for sid, child in list(self.children.items()):
                try:
                    child.conn.send({"cmd": "DETACH"})
                    _ = child.conn.recv()
                except Exception:
                    pass
                try:
                    child.proc.join(timeout=1.0)
                except Exception:
                    pass

        try:
            os.unlink(self.sock_path)
        except FileNotFoundError:
            pass

    def _spawn_service(self, flow_id: int, fd: int):
        service_id = self.next_service_id
        self.next_service_id += 1

        parent_conn, child_conn = self.ctx.Pipe()
        proc = self.ctx.Process(
            target=child_main,
            args=(child_conn, service_id, flow_id, fd),
            daemon=True,
        )

        t0 = time.perf_counter()
        proc.start()
        msg = parent_conn.recv()
        spawn_to_ready_s = time.perf_counter() - t0

        if msg.get("type") != "READY":
            raise RuntimeError(f"child failed to start: {msg}")

        child = ChildRef(service_id, proc, parent_conn, flow_id, fd)
        self.children[service_id] = child
        self.flow_to_service[flow_id] = service_id

        return service_id, spawn_to_ready_s

    def _reap_dead(self):
        dead = []
        for sid, child in self.children.items():
            if not child.proc.is_alive():
                dead.append(sid)
        for sid in dead:
            flow_id = self.children[sid].flow_id
            self.children.pop(sid, None)
            self.flow_to_service.pop(flow_id, None)

    def _handle_client(self, sock):
        try:
            req = _recv_json(sock)
            cmd = req.get("cmd")

            with self.lock:
                self._reap_dead()

                if cmd == "ATTACH":
                    flow_id = int(req["flow_id"])
                    fd = int(req.get("fd", -1))

                    if flow_id in self.flow_to_service:
                        sid = self.flow_to_service[flow_id]
                        _send_json(sock, {"ok": True, "service_id": sid, "already_attached": True})
                        return

                    sid, spawn_to_ready_s = self._spawn_service(flow_id, fd)
                    _send_json(
                        sock,
                        {
                            "ok": True,
                            "service_id": sid,
                            "preload_s": PRELOAD_INFO["preload_s"],
                            "spawn_to_ready_s": spawn_to_ready_s,
                        },
                    )
                    return

                elif cmd == "SET_CONTROL":
                    sid = int(req["service_id"])
                    control = bool(req["control"])
                    child = self.children.get(sid)
                    if child is None:
                        _send_json(sock, {"ok": False, "error": "bad_service"})
                        return
                    child.conn.send({"cmd": "SET_CONTROL", "control": control})
                    resp = child.conn.recv()
                    _send_json(sock, {"ok": resp.get("type") == "OK", "resp": resp})
                    return

                elif cmd == "DETACH":
                    sid = int(req["service_id"])
                    child = self.children.get(sid)
                    if child is None:
                        _send_json(sock, {"ok": False, "error": "bad_service"})
                        return
                    child.conn.send({"cmd": "DETACH"})
                    resp = child.conn.recv()
                    child.proc.join(timeout=1.0)
                    self.flow_to_service.pop(child.flow_id, None)
                    self.children.pop(sid, None)
                    _send_json(sock, {"ok": resp.get("type") == "OK", "resp": resp})
                    return

                elif cmd == "STATUS":
                    out = {}
                    for sid, child in self.children.items():
                        child.conn.send({"cmd": "STATUS"})
                    for sid, child in self.children.items():
                        out[sid] = child.conn.recv()
                    _send_json(
                        sock,
                        {
                            "ok": True,
                            "preload_s": PRELOAD_INFO["preload_s"],
                            "children": out,
                        },
                    )
                    return

                else:
                    _send_json(sock, {"ok": False, "error": f"unknown cmd {cmd}"})
        except Exception as e:
            _send_json(sock, {"ok": False, "error": str(e)})
        finally:
            try:
                sock.close()
            except Exception:
                pass


# -----------------------------------------------------------------------------
# Socket helpers
# -----------------------------------------------------------------------------

def _recv_json(sock):
    data = b""
    while not data.endswith(b"\n"):
        chunk = sock.recv(4096)
        if not chunk:
            raise RuntimeError("eof")
        data += chunk
    return json.loads(data.decode())


def _send_json(sock, obj):
    sock.sendall((json.dumps(obj) + "\n").encode())


def send_cmd(sock_path, obj):
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.connect(sock_path)
    _send_json(sock, obj)
    resp = _recv_json(sock)
    sock.close()
    return resp


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def cmd_broker(args):
    broker = Broker(args.sock)

    def _term(*_):
        broker.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, _term)
    signal.signal(signal.SIGTERM, _term)
    broker.start()


def cmd_attach(args):
    resp = send_cmd(args.sock, {"cmd": "ATTACH", "flow_id": args.flow_id, "fd": args.fd})
    print(json.dumps(resp, indent=2))


def cmd_control(args):
    resp = send_cmd(
        args.sock,
        {"cmd": "SET_CONTROL", "service_id": args.service_id, "control": bool(args.control)},
    )
    print(json.dumps(resp, indent=2))


def cmd_detach(args):
    resp = send_cmd(args.sock, {"cmd": "DETACH", "service_id": args.service_id})
    print(json.dumps(resp, indent=2))


def cmd_status(args):
    resp = send_cmd(args.sock, {"cmd": "STATUS"})
    print(json.dumps(resp, indent=2))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)

    p = sub.add_parser("broker")
    p.add_argument("--sock", default="/tmp/astraea_prefork_demo.sock")
    p.set_defaults(func=cmd_broker)

    p = sub.add_parser("attach")
    p.add_argument("--sock", default="/tmp/astraea_prefork_demo.sock")
    p.add_argument("--flow-id", type=int, required=True)
    p.add_argument("--fd", type=int, default=-1)
    p.set_defaults(func=cmd_attach)

    p = sub.add_parser("control")
    p.add_argument("--sock", default="/tmp/astraea_prefork_demo.sock")
    p.add_argument("--service-id", type=int, required=True)
    p.add_argument("--control", type=int, choices=[0, 1], required=True)
    p.set_defaults(func=cmd_control)

    p = sub.add_parser("detach")
    p.add_argument("--sock", default="/tmp/astraea_prefork_demo.sock")
    p.add_argument("--service-id", type=int, required=True)
    p.set_defaults(func=cmd_detach)

    p = sub.add_parser("status")
    p.add_argument("--sock", default="/tmp/astraea_prefork_demo.sock")
    p.set_defaults(func=cmd_status)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
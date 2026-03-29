#!/usr/bin/env python3
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
import os
import time
import math
import threading
from os import path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from astraea.agent.agent import Agent
from astraea.agent.definitions import transform_state, STATE_DIM, ACTION_DIM, GLOBAL_DIM
from astraea.helpers.utils import Params
import tcp_sockopt


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

config_path = path.abspath(
    path.join(path.dirname(__file__), "astraea", "astraea.json")
)
model_path = path.abspath(path.join(path.dirname(__file__), "models", "exported"))


def map_action(action, cwnd):
    if action >= 0:
        out = 1 + 0.025 * action
        out = math.ceil(out * cwnd)
    else:
        out = 1 / (1 - 0.025 * action)
        out = math.floor(out * cwnd)
    if out < 1:
        out = 1
    return int(out)


def get_action_info():
    action_scale = np.array([1.0])
    action_range = (-action_scale, action_scale)
    return action_scale, action_range


def inference(agent, state, s0_rec_buffer_inf=None):
    s0, _ = transform_state(state)

    if s0_rec_buffer_inf is None:
        s0_rec_buffer_inf = np.zeros(agent.s_dim, dtype=np.float32)

    s0_rec_buffer_inf = np.concatenate((s0_rec_buffer_inf[len(s0):], s0))

    a = agent.get_action(s0_rec_buffer_inf, False)
    a = float(a[0][0][0])

    cwnd_now = int(state["cwnd"])
    out_cwnd = map_action(a, cwnd_now)

    reply = {"cwnd": int(out_cwnd)}
    return reply, s0_rec_buffer_inf


def make_agent(model_path, params, s_dim, s_dim_global, a_dim, action_scale, action_range):
    agent = Agent(
        s_dim,
        s_dim_global,
        a_dim,
        batch_size=params.dict["batch_size"],
        h1_shape=params.dict["h1_shape"],
        h2_shape=params.dict["h2_shape"],
        stddev=0.05,
        policy_delay=params.dict["policy_delay"],
        mem_size=params.dict["memsize"],
        gamma=params.dict["gamma"],
        lr_c=params.dict["lr_c"],
        lr_a=params.dict["lr_a"],
        tau=params.dict["tau"],
        PER=params.dict["PER"],
        LOSS_TYPE=params.dict["LOSS_TYPE"],
        noise_type=3,
        noise_exp=params.dict["noise_exp"],
        train_exp=params.dict["train_exp"],
        action_scale=action_scale,
        action_range=action_range,
        is_global=params.dict["global"],
        ckpt_dir=model_path,
    )

    eval_sess = tf.Session()
    agent.assign_sess(eval_sess)
    agent.load_model()
    return agent


class AstraeaService:
    def __init__(self, config, model_path):
        tf.logging.set_verbosity(tf.logging.ERROR)

        action_scale, action_range = get_action_info()
        self.params = Params(config)

        s_dim, a_dim, s_dim_global = STATE_DIM, ACTION_DIM, GLOBAL_DIM
        single_dim = s_dim
        if self.params.dict["recurrent"]:
            s_dim = single_dim * self.params.dict["rec_dim"]

        self.agent = make_agent(
            model_path,
            self.params,
            s_dim,
            s_dim_global,
            a_dim,
            action_scale,
            action_range,
        )

        self.interval_sec = 0.020
        self.lock = threading.RLock()

        # Single-flow state
        self.flow_id = None
        self.fd = None
        self.control = False
        self.rec_buf = None
        self.meta = None
        self.thread = None
        self.started_at = None
        self.stop_event = threading.Event()

    def _now_us(self):
        return int(time.monotonic() * 1_000_000)

    def _get_state(self):
        info = tcp_sockopt.get_tcp_deepcc_info(self.fd)
        now_us = self._now_us()

        if self.meta is None:
            time_delta = 1
            max_tput = int(info.get("avg_thr", 0))
        else:
            time_delta = max(1, now_us - self.meta["last_ts_us"])
            max_tput = max(self.meta["max_tput"], int(info.get("avg_thr", 0)))

        loss_bytes = int(info.get("loss_bytes", 0))
        loss_ratio = int((loss_bytes * 1_000_000) / max(1, time_delta))

        info["max_tput"] = max_tput
        info["loss_ratio"] = loss_ratio
        info["time_delta"] = time_delta

        self.meta = {
            "last_ts_us": now_us,
            "max_tput": max_tput,
        }
        return info

    def _cleanup_locked(self):
        fd = self.fd

        self.flow_id = None
        self.fd = None
        self.control = False
        self.rec_buf = None
        self.meta = None
        self.thread = None
        self.started_at = None
        self.stop_event = threading.Event()

        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass

    def _run_flow(self):
        while not self.stop_event.is_set():
            with self.lock:
                if self.fd is None:
                    return
                fd = self.fd
                flow_id = self.flow_id
                control = self.control
                rec_buf = self.rec_buf
                started_at = self.started_at

            try:
                state = self._get_state()
            except Exception as e:
                print(f"[py] flow={flow_id} get_state failed: {e}", flush=True)
                with self.lock:
                    self._cleanup_locked()
                return

            try:
                reply, rec_buf = inference(
                    self.agent,
                    state,
                    s0_rec_buffer_inf=rec_buf,
                )

                if control:
                    tcp_sockopt.set_cwnd(fd, int(reply["cwnd"]))

                with self.lock:
                    if self.fd is None:
                        return
                    self.rec_buf = rec_buf

                elapsed_s = None if started_at is None else (time.monotonic() - started_at)
                t_str = "off" if elapsed_s is None else f"{elapsed_s:.3f}s"

                print(
                    f"[py] flow={flow_id} t={t_str} control={control} "
                    f"in_cwnd={state.get('cwnd')} "
                    f"avg_thr={state.get('avg_thr')} "
                    f"min_rtt={state.get('min_rtt')} "
                    f"out={reply}",
                    flush=True,
                )
            except Exception as e:
                print(f"[py] flow={flow_id} step failed: {e}", flush=True)
                with self.lock:
                    self._cleanup_locked()
                return

            time.sleep(self.interval_sec)

    def attach_flow(self, flow_id, fd):
        with self.lock:
            if self.fd is not None:
                if self.flow_id == flow_id and self.fd == int(fd):
                    return {"ok": True, "already_exists": True}
                return {"ok": False, "error": "service_busy"}

            self.flow_id = int(flow_id)
            self.fd = int(fd)
            self.control = False
            self.rec_buf = None
            self.meta = None
            self.started_at = None
            self.stop_event.clear()

            th = threading.Thread(target=self._run_flow, daemon=True)
            self.thread = th
            th.start()

        print(f"[py] attached flow={flow_id} fd={fd}", flush=True)
        return {"ok": True}

    def set_control(self, flow_id, control):
        with self.lock:
            if self.fd is None or self.flow_id != int(flow_id):
                return {"ok": False, "error": "unknown_flow"}

            control = bool(control)

            if control and not self.control:
                self.started_at = time.monotonic()
            elif not control:
                self.started_at = None

            self.control = control

        print(f"[py] flow={flow_id} control set to {control}", flush=True)
        return {"ok": True, "control": control}

    def detach_flow(self, flow_id):
        with self.lock:
            if self.fd is None or self.flow_id != int(flow_id):
                return {"ok": False, "error": "unknown_flow"}

            self.stop_event.set()
            self._cleanup_locked()

        print(f"[py] detached flow={flow_id}", flush=True)
        return {"ok": True}


if __name__ == "__main__":
    flow_id = int(os.environ["ASTRAEA_FLOW_ID"])
    fd = int(os.environ["ASTRAEA_FLOW_FD"])
    config = os.environ["ASTRAEA_CONFIG"]
    model = os.environ["ASTRAEA_MODEL"]

    svc = AstraeaService(config, model)
    svc.attach_flow(flow_id, fd)
    svc.set_control(flow_id, True)

    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        svc.detach_flow(flow_id)
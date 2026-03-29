#!/usr/bin/env python3.7
import warnings
warnings.filterwarnings("ignore", category=FutureWarning) # supress FutureWarning
import argparse
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress INFO and WARNING logs
from os import path
import numpy as np
import tensorflow as tf
import json
import signal
import math
import matplotlib
import csv

matplotlib.use("Agg")
from matplotlib import pyplot as plt
import time

import context

from agent.agent import Agent
from agent.definitions import transform_state, STATE_DIM, ACTION_DIM, GLOBAL_DIM
from helpers.utils import Params
from helpers.logger import logger
from helpers.ipc_socket import IPCSocket

config_path = path.abspath(
    path.join(path.dirname(__file__), "astraea.json")
)
model_path = path.abspath(path.join(path.dirname(__file__), os.pardir, "model"))


def map_action(action, cwnd):
    if action >= 0:
        out = 1 + 0.025 * (action)
        out = math.ceil(out * cwnd)
    else:
        out = 1 / (1 - 0.025 * (action))
        out = math.floor(out * cwnd)
    return out

def get_action_info():
    action_scale = np.array([1.0])
    action_range = (-action_scale, action_scale)
    return action_scale, action_range


def inference(flow_id, agent, state, s0_rec_buffer_inf=None):
    # logger.info("inference start: flow_id: {}, step: {} , state: {}".format(flow_id, step, state))
    s0, _ = transform_state(state)
    if s0_rec_buffer_inf is None:
        s0_rec_buffer_inf = np.zeros(agent.s_dim)
    s0_rec_buffer_inf = np.concatenate((s0_rec_buffer_inf[len(s0) :], s0))
    a = agent.get_action(s0_rec_buffer_inf, False)
    a = a[0][0][0]
    a = map_action(a, state["cwnd"])
    return a, s0_rec_buffer_inf


def make_agent(
    model_path, params, s_dim, s_dim_global, a_dim, action_scale, action_range
):
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
    # init tf SingularMonitoredSession
    eval_sess = tf.compat.v1.Session()
    # load model
    agent.assign_sess(eval_sess)
    agent.load_model()
    return agent


def plot_cwnd_history(prefix):
    data = np.loadtxt("{}.log".format(prefix))
    plt.figure()
    plt.plot(data[:, 0], data[:, 1])
    plt.title("CWND")
    plt.savefig("{}.png".format(prefix))
    sys.stderr.write("Astraea RL helper: ploted cwnd....\n")


def main():
    cwnd_history = []
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    global agent
    global params

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default=config_path, help="configuration file"
    )
    parser.add_argument(
        "--model-path", type=str, default=model_path, help="path of saved models"
    )
    parser.add_argument(
        "--ipc-path", type=str, help="IPC path of communication", required=True
    )

    args = parser.parse_args()
    action_scale, action_range = get_action_info()
    params = Params(args.config)

    #sys.stderr.write(f"PyHelper: Loading model from: {args.model_path}\n")

    s_dim, a_dim, s_dim_global = STATE_DIM, ACTION_DIM, GLOBAL_DIM
    single_dim = s_dim
    if params.dict["recurrent"]:
        s_dim = single_dim * params.dict["rec_dim"]
    agent = make_agent(
        args.model_path, params, s_dim, s_dim_global, a_dim, action_scale, action_range
    )
    s0_rec_buffer_inf = np.zeros(s_dim)

    # connect to C++ flow module
    ipc_sock = IPCSocket()
    ipc_sock.connect(args.ipc_path)

    while True:
        info = ipc_sock.read()
        info = json.loads(info)
        state = info["state"]
        # logger.info("RL: state is {}".format(state))
        act, s0_rec_buffer_inf = inference(
            info["flow_id"], agent, state, s0_rec_buffer_inf=s0_rec_buffer_inf
        )
        reply = {}
        reply["cwnd"] = act
        # logger.info("RL: action is {}".format(act))
        ipc_sock.write(json.dumps(reply))


if __name__ == "__main__":
    main()

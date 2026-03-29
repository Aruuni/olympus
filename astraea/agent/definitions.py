import math
import numpy as np
import random
from . import context
import random
from datetime import datetime
from helpers.logger import logger

STATE_DIM = 10
ACTION_DIM = 1
GLOBAL_DIM = 12

def is_global_state(state):
    if type(list(state.values())[0]) == dict:
        return True
    else:
        return False


def transform_state(state, env=None, flow_id=None):
    # return a array with transformed states
    def state_dict_2_array(state_dict):
        state = []

        if state_dict["avg_thr"] == 0:
            state.append(0.5)
        else:
            state.append(
                state_dict["avg_thr"] / state_dict["max_tput"]
                if state_dict["max_tput"] > 0
                else 0
            )
        if state_dict["avg_urtt"] == 0:
            state.append(2)
        elif state_dict["min_rtt"] == 0:
            state.append(0)
        else:
            state.append(state_dict["avg_urtt"] / state_dict["min_rtt"])

        if state_dict["srtt_us"] == 0:
            state.append(2)
        elif state_dict["min_rtt"] == 0:
            state.append(0)
        else:
            state.append(state_dict["srtt_us"] / 8 / state_dict["min_rtt"])

        if state_dict["min_rtt"] == 0 or state_dict["max_tput"] == 0:
            state.append(0)
        else:
            state.append(
                state_dict["cwnd"]
                * 1460
                * 8
                / (state_dict["min_rtt"] / 1e6)
                / state_dict["max_tput"]
                / 10
            )
            # state.append(state_dict["cwnd"] / 1000 /  (state_dict["max_tput"] / 1e7  *  state_dict["min_rtt"] / 1e5))
        state.append(state_dict["max_tput"] / 1e7)
        state.append(state_dict["min_rtt"] / 5e5)
        state.append(
            state_dict["loss_ratio"] / state_dict["max_tput"]
            if state_dict["max_tput"] > 0
            else 0
        )
        state.append(state_dict["packets_out"] / state_dict["cwnd"])
        state.append(
            state_dict["pacing_rate"] / state_dict["max_tput"]
            if state_dict["max_tput"] > 0
            else 0
        )
        state.append(state_dict["retrans_out"] / state_dict["cwnd"])

        if state[2] > 2:
            state[2] = 2
        if state[1] > 2:
            state[1] = 2
        if state[3] > 2:
            state[3] = 2
        if state[8] > 2:
            state[8] = 2

        return state

    if is_global_state(state):
        assert env is not None
        thr = np.sum([state[s]["avg_thr"] / 5e7 for s in state.keys()])
        min_thr = np.min([state[s]["avg_thr"] / 5e7 for s in state.keys()])
        max_thr = np.max([state[s]["avg_thr"] / 5e7 for s in state.keys()])
        lat = np.mean([state[s]["avg_urtt"] / 5e5 for s in state.keys()])
        min_window = np.min([state[s]["cwnd"] / 1000 for s in state.keys()])
        max_window = np.max([state[s]["cwnd"] / 1000 for s in state.keys()])
        mean_window = np.mean([state[s]["cwnd"] / 1000 for s in state.keys()])
        loss = np.mean([state[s]["loss_ratio"] / 1e6 for s in state.keys()])
        num_flows = len(state.keys())
        global_state = [
            thr,
            min_thr,
            max_thr,
            lat,
            min_window,
            max_window,
            mean_window,
            loss,
            num_flows / 10,
            env.world.one_way_delay / 500.0,
            env.world.bdp / 10,
            env.world.bandwidth / 500,
        ]
        return state_dict_2_array(state[flow_id]), global_state
    else:
        return state_dict_2_array(state), []


def max_action(env):
    max_window = (
        env.world.bandwidth * 1e6 * (2 * env.world.one_way_delay / 1e3) / (1460 * 8)
    )
    return int(max_window * 1.5)


def reverse_action(action, cwnd):
    if action == 0:
        return 1
    action /= cwnd
    if action >= 1:
        out = (action - 1) * 40
    else:
        out = (1 - 1 / action) * 40
    if out < -1:
        out = -1
    if out > 1:
        out = 1
    return out

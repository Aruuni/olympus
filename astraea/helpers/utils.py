import os
from os import path
from shutil import Error
import sys
import socket
import signal
import errno
import json
from traceback import print_exception
import traceback
from numpy.core.fromnumeric import trace
import yaml
import subprocess
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import random
import copy

np.set_printoptions(threshold=np.inf)
from operator import add
from . import context
from helpers.subprocess_wrappers import check_call, check_output, call
from helpers.logger import logger

def exception_handler(func):
    def inner_function(*args, **kwargs):
        try:
            func(*args, **kwargs)
        except Exception as e:
            sys.stderr.write(traceback.format_exec())
            print_exception(e)

    return inner_function


def create_input_op_shape(obs, tensor):
    input_shape = [x or -1 for x in tensor.shape.as_list()]
    return np.reshape(obs, input_shape)


class Params:
    def __init__(self, json_path):
        self.update(json_path)

    def save(self, json_path):
        with open(json_path, "w") as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        return self.__dict__


def sample_action(alg, state, tun_id=None):
    if tun_id == 1:
        return 200
    else:
        return 400


def check_kernel_module(cc_name):
    if call("sudo modprobe tcp_{}".format(cc_name), shell=True) != 0:
        sys.exit("kernel module tcp_{} is not available".format(cc_name))


def make_sure_dir_exists(d):
    try:
        os.makedirs(d)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise


tmp_dir = path.join(context.base_dir, "tmp")
make_sure_dir_exists(tmp_dir)

def update_submodules():
    cmd = "git submodule update --init --recursive"
    check_call(cmd, shell=True)
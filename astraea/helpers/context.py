import os
from os import path
import sys
src_dir = path.abspath(path.join(path.dirname(__file__), os.pardir))
helper_dir = path.abspath(path.join(path.dirname(__file__)))
sys.path.append(src_dir)
sys.path.append(helper_dir)

log_dir = path.abspath(path.join(src_dir, os.pardir, 'log'))
# project dir
base_dir = path.abspath(path.join(src_dir, os.pardir)) 
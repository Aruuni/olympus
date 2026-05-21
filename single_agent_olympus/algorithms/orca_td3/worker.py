"""Legacy alias for the renamed td3 worker."""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ALG_DIR = os.path.dirname(_HERE)
_PKG = os.path.dirname(_ALG_DIR)
_REPO = os.path.dirname(_PKG)
sys.path.insert(0, _REPO)

from single_agent_olympus.algorithms.td3.worker import run


if __name__ == '__main__':
    run()

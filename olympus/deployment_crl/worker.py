"""Adapt an astraea_listener socket process to the Olympus SAC worker."""

import os
import re
import sys
import time


_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def _safe_id(value) -> str:
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', str(value or 'unknown'))


def prepare_worker_environment() -> None:
    """Translate generic deployment-listener variables to SAC conventions."""
    aliases = {
        'OC_FLOW_FD': 'ASTRAEA_FLOW_FD',
        'OC_FLOW_ID': 'ASTRAEA_FLOW_ID',
        'SAO_CONFIG': 'ASTRAEA_CONFIG',
        'SAO_CHECKPOINT': 'ASTRAEA_MODEL',
    }
    for target, source in aliases.items():
        if not os.environ.get(target) and os.environ.get(source):
            os.environ[target] = os.environ[source]

    flow_id = os.environ.get('OC_FLOW_ID', '0')
    os.environ.setdefault('OC_CPORT', flow_id)
    os.environ.setdefault('OC_STATE_FD', '-1')
    os.environ.setdefault('SAO_REQUIRE_CHECKPOINT', '1')
    os.environ.setdefault('SAO_EPISODE', os.environ.get(
        'OLYMPUS_CRL_EPOCH', str(int(time.time()))))

    trace_dir = os.environ.get('OLYMPUS_TRACE_DIR', '')
    if trace_dir:
        os.makedirs(trace_dir, exist_ok=True)
        os.environ['SAO_TRACE_LOG'] = os.path.join(
            trace_dir,
            f'sac_crl_flow_{_safe_id(flow_id)}_pid_{os.getpid()}.csv',
        )


def main() -> None:
    prepare_worker_environment()
    from olympus.algorithms.sac.worker import run
    run()


if __name__ == '__main__':
    main()

"""Backend facade for worker flow I/O.

Workers use this module instead of importing ``tcp_sockopt`` directly. The
default backend delegates to Linux TCP socket helpers. The RayNet backend
connects to a per-episode virtual-flow service that presents simulation agents
with the same ``get_tcp_deepcc_info`` / ``set_cwnd`` shape.
"""

import os
import time
from multiprocessing.managers import BaseManager


class _RayNetManager(BaseManager):
    pass


_RayNetManager.register('flow_service')
_RAYNET_SERVICE = None


def _backend_name():
    return str(os.environ.get('OC_FLOW_BACKEND', 'tcp')).strip().lower()


def _tcp_sockopt():
    import tcp_sockopt
    return tcp_sockopt


def _raynet_service():
    global _RAYNET_SERVICE
    if _RAYNET_SERVICE is not None:
        return _RAYNET_SERVICE

    address = os.environ.get('OC_RAYNET_FLOW_ADDR', '')
    key = os.environ.get('OC_RAYNET_FLOW_KEY', '')
    if not address or not key:
        raise RuntimeError('RayNet flow backend missing OC_RAYNET_FLOW_ADDR/KEY')

    host, port = address.rsplit(':', 1)
    manager = _RayNetManager(
        address=(host, int(port)),
        authkey=bytes.fromhex(key),
    )
    deadline = time.monotonic() + 30.0
    last_error = None
    while time.monotonic() < deadline:
        try:
            manager.connect()
            _RAYNET_SERVICE = manager.flow_service()
            return _RAYNET_SERVICE
        except (ConnectionRefusedError, FileNotFoundError, OSError) as exc:
            last_error = exc
            time.sleep(0.05)
    if last_error is not None:
        raise last_error
    manager.connect()
    _RAYNET_SERVICE = manager.flow_service()
    return _RAYNET_SERVICE


def get_tcp_deepcc_info(flow_fd):
    """Return one raw metrics dictionary for the selected flow backend."""
    if _backend_name() == 'raynet':
        flow_id = int(os.environ.get('OC_FLOW_ID', flow_fd))
        return dict(_raynet_service().get_tcp_deepcc_info(flow_id))
    return _tcp_sockopt().get_tcp_deepcc_info(flow_fd)


def set_cwnd(flow_fd, cwnd):
    """Apply an absolute CWND request through the selected flow backend."""
    if _backend_name() == 'raynet':
        flow_id = int(os.environ.get('OC_FLOW_ID', flow_fd))
        return _raynet_service().set_cwnd(flow_id, int(cwnd))
    return _tcp_sockopt().set_cwnd(flow_fd, cwnd)

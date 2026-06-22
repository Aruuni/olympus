"""RayNet simulation backend for Olympus.

RayNet is not an emulated TCP backend: it does not create namespaces, cports,
iperf flows, or kernel sockets. The orchestrator dispatches RayNet episodes to
``olympus.environments.raynet.runner`` instead of the Mininet listener path.
This class exists so environment selection and backend validation remain
consistent with the rest of Olympus.
"""

import os
from pathlib import Path

from olympus.environments.base import NetworkEnv


class RaynetEnv(NetworkEnv):
    """Lightweight handle for RayNet simulation episodes."""

    def __init__(self, n=1, bw=10, delay=20, qsize=None, bdp_mult=1.0,
                 loss=None, duration=60, cport=11111, cc_algo='orca',
                 instance_id=None, unique_cports=False, per_flow_delays=None,
                 raynet_path=None, ini_path=None, section='General',
                 protocol='orca', raynet_runner=None, **extra):
        self.n = int(n)
        self.bw = float(bw)
        self.delay = float(delay)
        self.qsize = qsize
        self.bdp_mult = bdp_mult
        self.loss = loss
        self.duration = duration
        self.cport = cport
        self.cc_algo = cc_algo
        self.instance_id = instance_id
        self.unique_cports = unique_cports
        self.per_flow_delays = per_flow_delays
        self.protocol = str(protocol or 'orca').lower()
        self.section = str(section or 'General')
        self.ini_path = ini_path
        self.raynet_path = Path(
            raynet_path
            or extra.get('raynet_root')
            or os.environ.get('RAYNET_PATH', '/home/james/raynet')
        ).expanduser()
        self.raynet_runner = Path(
            raynet_runner
            or extra.get('runner')
            or self.raynet_path / 'runners' / 'olympus_runner.sh'
        ).expanduser()
        self.started = False

    def start(self) -> None:
        if self.protocol != 'orca':
            raise ValueError(
                f'RayNet backend v1 supports protocol="orca" only, got {self.protocol!r}')
        if not self.raynet_path.exists():
            raise FileNotFoundError(f'RayNet path not found: {self.raynet_path}')
        build_dir = self.raynet_path / 'build'
        if not build_dir.exists():
            raise FileNotFoundError(
                f'RayNet build directory not found: {build_dir}. Build RayNet with ./build.sh.')
        if not self.raynet_runner.exists():
            raise FileNotFoundError(f'RayNet Olympus runner not found: {self.raynet_runner}')
        if self.ini_path is not None and not Path(self.ini_path).expanduser().exists():
            raise FileNotFoundError(f'RayNet ini_path not found: {self.ini_path}')
        self.started = True

    def set_link(self, bw=None, delay=None, loss=None) -> None:
        if bw is not None:
            self.bw = float(bw)
        if delay is not None:
            self.delay = float(delay)
        if loss is not None:
            self.loss = loss
        print('[raynet-env] set_link ignored; RayNet scenarios are fixed by INI/scenario files',
              flush=True)

    def run_iperf(self, monitor_interval=0.1, start_delays=None,
                  flow_durations=None) -> None:
        raise RuntimeError(
            'RayNet simulation backend does not run iperf; use the RayNet episode dispatcher')

    def stop(self) -> None:
        self.started = False


ENV_CLASS = RaynetEnv

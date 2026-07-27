"""Smoke-test the rebuilt RayNet stack via the production olympus_runner protocol.

Mirrors what olympus/environments/raynet/env.py sends: a start message with an
episode dict rendered from base_environment.ini, then a few steps.
"""
import json
import socket
import subprocess
import sys
from pathlib import Path

SIM = Path('/its/home/mm2350/extra/Olympusv2/olympus/environments/raynet/sim')
RAYNET = SIM / 'raynet'
OMNET = SIM / 'omnetpp'

BW = 10.0        # Mbps
DELAY = 20.0     # ms RTT
INTERVAL_S = 0.02
DURATION_S = 2.0

episode = {
    'protocol': 'DeepCC',
    'ini_path': str(RAYNET / '_environments' / 'base_environment.ini'),
    'section': 'General',
    'bw': BW,
    'delay': DELAY,
    'flows': 1,
    'bdp_mult': 4,
    'duration': DURATION_S,
    'interval_ms': INTERVAL_S * 1000.0,
    'quiet': True,
    'omnet_path': str(OMNET),
    'replacements': {
        'raynet_path': str(RAYNET),
        'omnet_path': str(OMNET),
        'inet': str(OMNET / 'samples' / 'inet4.5'),
        'protocol': 'DeepCC',
        'cc_algo': 'TcpPacedNoCC',
        'bw': f'{BW:.12g}Mbps',
        'delay': f'{DELAY / 2.0:.12g}ms',
        'qsize': f"{round(BW * DELAY * 1000.0 * 4)}b",
        'max_rl_steps': str(int(round(DURATION_S / INTERVAL_S))),
    },
    'overrides': {
        '**.numberOfFlows': '1',
        '**.fixedIntervalDuration': f'{INTERVAL_S:.12g}',
        '**.step_duration': f'{INTERVAL_S:.12g}s',
    },
}

parent, child = socket.socketpair()
proc = subprocess.Popen(
    [str(RAYNET / 'runners' / 'olympus_runner.sh'),
     '--control-fd', str(child.fileno())],
    pass_fds=(child.fileno(),),
    env={'PATH': '/usr/bin:/bin', 'HOME': str(Path.home()),
         'OMNET_PATH': str(OMNET), 'RAYNET_PATH': str(RAYNET)},
)
child.close()
reader = parent.makefile('r', encoding='utf-8', newline='\n')
writer = parent.makefile('w', encoding='utf-8', newline='\n')


def send(msg):
    writer.write(json.dumps(msg) + '\n')
    writer.flush()


def recv():
    line = reader.readline()
    if not line:
        raise RuntimeError(f'runner closed (rc={proc.poll()})')
    return json.loads(line)


send({'type': 'start', 'episode': episode})
reset = recv()
assert reset['type'] == 'reset', reset
obs = reset['observations']
print(f"RESET ok: flows={list(obs)} fields/flow="
      f"{sorted(next(iter(obs.values())))[:8] if obs and isinstance(next(iter(obs.values())), dict) else type(next(iter(obs.values()))).__name__ if obs else 'none'}")

for i in range(10):
    send({'type': 'step', 'actions': {}})
    step = recv()
    info = step.get('info', {})
    if i in (0, 4, 9):
        print(f"STEP {i}: t={info.get('time_s'):.3f}s rewards={step.get('rewards')} "
              f"terminated={step.get('terminateds', {}).get('__all__')}")
    if step.get('terminateds', {}).get('__all__') or info.get('simDone'):
        print('sim ended early at step', i)
        break

reader.close()
writer.close()
parent.close()
proc.wait(timeout=30)
print('SMOKE OK, runner exit code:', proc.returncode)

"""Launch a persistent online-learning SAC service and per-flow workers."""

import argparse
import copy
import fcntl
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import yaml


_ROOT = Path(__file__).resolve().parents[2]


def _acquire_service_lock(path):
    handle = open(path, 'a+')
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        raise SystemExit(
            f'another Olympus deployment service holds {path}; stop it first')
    return handle


def _selected_blocks(cfg: dict) -> tuple:
    selected = ((cfg.get('algorithms') or {}).get('sac') or {})
    training = dict(selected.get('training') or {})
    training.update(dict(cfg.get('training') or {}))
    agent = dict(selected.get('agent') or {})
    agent.update(dict(cfg.get('agent') or {}))
    return training, agent


def _checkpoint_path(cfg: dict, override: str = None) -> str:
    training, _ = _selected_blocks(cfg)
    deployment = cfg.get('deployment') or {}
    return str(override or deployment.get('checkpoint')
               or training.get('resume_from') or training.get('checkpoint') or '')


def prepare_service_config(cfg: dict, source_checkpoint: str,
                           state_dir: str) -> tuple:
    """Write a resolved service config and return its paths.

    An existing rolling checkpoint wins over the source checkpoint, making a
    service restart a true continuation.  The source is copied only on the
    first launch so the learner can always load its ordinary ``checkpoint``
    path and atomically replace it thereafter.
    """
    cfg = copy.deepcopy(cfg)
    runtime = cfg.setdefault('runtime', {})
    if str(runtime.get('algorithm', '')) != 'sac':
        raise ValueError('continual deployment requires runtime.algorithm: sac')
    reward_name = str(runtime.get('reward', ''))
    if reward_name not in ('proteus', 'crl_network_utility'):
        raise ValueError(
            'continual deployment requires runtime.reward: proteus')
    # Accept the former public name, but persist and launch the canonical one.
    runtime['reward'] = 'proteus'

    state_dir = os.path.abspath(state_dir)
    os.makedirs(state_dir, exist_ok=True)
    live_checkpoint = os.path.join(state_dir, 'sac_continual.pt')
    if not os.path.exists(live_checkpoint):
        shutil.copy2(source_checkpoint, live_checkpoint)
        envelope = source_checkpoint + '.envelope.json'
        if os.path.exists(envelope):
            shutil.copy2(envelope, live_checkpoint + '.envelope.json')

    training, agent = _selected_blocks(cfg)
    training.update({
        'resume_from': None,
        'checkpoint': live_checkpoint,
        'log_path': os.path.join(state_dir, 'learner_metrics.csv'),
    })
    cfg['training'] = training
    cfg['agent'] = agent

    deployment = cfg.setdefault('deployment', {})
    selected_continual = dict(cfg.get('continual') or {})
    selected_continual.update(dict(deployment.get('continual') or {}))
    defaults = {
        'enabled': True,
        'learner_port': 6401,
        'updates_per_transition': 0.25,
        'max_update_burst': 100,
        'persist_replay': True,
        'replay_save_every_s': 300.0,
        'replay_path': os.path.join(state_dir, 'sac_replay.pkl'),
        'exploration': False,
        'weight_pull_every': 50,
        'worker_push_every': 16,
        'distillation': {
            'enabled': True,
            'actor_weight': 0.05,
            'critic_weight': 0.05,
            'anchor_capacity': 2048,
            'anchor_min_sequences': 64,
            'batch_sequences': 8,
            'capture_every_updates': 25,
            # Zero keeps the original trusted deployment checkpoint frozen.
            'teacher_update_every': 0,
        },
    }
    defaults.update(selected_continual)
    defaults['enabled'] = True
    defaults['state_dir'] = state_dir
    defaults['replay_path'] = os.path.abspath(defaults['replay_path'])
    cfg['continual'] = defaults
    deployment['continual'] = defaults
    deployment['checkpoint'] = live_checkpoint

    resolved = os.path.join(state_dir, 'config.resolved.yaml')
    tmp = f'{resolved}.tmp.{os.getpid()}'
    try:
        with open(tmp, 'w') as handle:
            yaml.safe_dump(cfg, handle, sort_keys=False)
        os.replace(tmp, resolved)
    finally:
        try:
            if os.path.exists(tmp):
                os.unlink(tmp)
        except OSError:
            pass
    return cfg, resolved, live_checkpoint


def _forward_output(proc, prefix):
    for line in proc.stdout:
        print(prefix + line, end='', flush=True)


def _start_learner(cfg, config_path, root):
    continual = cfg['continual']
    port = int(continual.get('learner_port', 6401))
    runtime = cfg.get('runtime') or {}
    action_name = str(runtime.get('action', 'cwnd_multiplier'))
    action_options = (cfg.get('actions') or {}).get(action_name) or {}
    env = dict(os.environ)
    env.update({
        'SAO_CONFIG': config_path,
        'OC_CONFIG': config_path,
        'SAO_STATE': str(runtime.get('state', 'min_rtt')),
        'OC_STATE': str(runtime.get('state', 'min_rtt')),
        'SAO_ACTION_NAME': action_name,
        'SAO_ACTION_CONFIG': json.dumps(action_options),
    })
    command = [
        sys.executable,
        str(root / 'olympus' / 'algorithms' / 'sac' / 'learner.py'),
        '--config', config_path,
        '--port', str(port),
    ]
    proc = subprocess.Popen(
        command, cwd=str(root), env=env, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, bufsize=1,
        start_new_session=True)
    key = ''
    ready = False
    for line in proc.stdout:
        print('[crl-learner] ' + line, end='', flush=True)
        if line.startswith('SAO_MANAGER_KEY='):
            key = line.strip().split('=', 1)[1]
        if '[learner] manager on port' in line:
            ready = True
            break
        if proc.poll() is not None:
            break
    if not key or not ready:
        code = proc.poll()
        if code is None:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait()
        raise RuntimeError(
            f'SAC continual learner failed during startup (exit={code})')
    threading.Thread(
        target=_forward_output, args=(proc, '[crl-learner] '), daemon=True
    ).start()
    return proc, f'127.0.0.1:{port}', key


def _listener_environment(cfg, config_path, checkpoint, manager_addr,
                          manager_key, trace_dir):
    runtime = cfg.get('runtime') or {}
    agent = cfg.get('agent') or {}
    continual = cfg.get('continual') or {}
    action_name = str(runtime.get('action', 'cwnd_multiplier'))
    action_options = (cfg.get('actions') or {}).get(action_name) or {}
    inherited_pythonpath = os.environ.get('PYTHONPATH', '')
    pythonpath = (str(_ROOT) if not inherited_pythonpath else
                  str(_ROOT) + os.pathsep + inherited_pythonpath)
    deterministic = not bool(continual.get('exploration', False))
    return dict(os.environ, **{
        'ASTRAEA_PYTHON': sys.executable,
        'ASTRAEA_CONFIG': config_path,
        'ASTRAEA_MODEL': checkpoint,
        'SAO_CONFIG': config_path,
        'OC_CONFIG': config_path,
        'SAO_CHECKPOINT': checkpoint,
        'SAO_MANAGER_ADDR': manager_addr,
        'SAO_MANAGER_KEY': manager_key,
        'SAO_ALGORITHM': 'sac',
        'SAO_REWARD': 'proteus',
        'SAO_STATE': str(runtime.get('state', 'min_rtt')),
        'OC_STATE': str(runtime.get('state', 'min_rtt')),
        'SAO_ACTION_NAME': action_name,
        'SAO_ACTION_CONFIG': json.dumps(action_options),
        'SAO_INTERVAL_MS': str(float(agent.get('interval_ms', 20))),
        'SAO_CWND_MIN': str(int(agent.get('cwnd_min', 10))),
        'SAO_CWND_MAX': str(int(agent.get('cwnd_max', 10000))),
        'SAO_HIDDEN': str(int(agent.get('hidden', 128))),
        'SAO_HEAD_HIDDEN': ('' if agent.get('head_hidden') is None
                            else str(int(agent['head_hidden']))),
        'SAO_DEAD_FLOW_MS': str(int(agent.get('dead_flow_ms', 1000))),
        'SAO_DETERMINISTIC': '1' if deterministic else '0',
        'SAO_REQUIRE_CHECKPOINT': '1',
        'SAO_WEIGHT_PULL_EVERY': str(int(
            continual.get('weight_pull_every', 50))),
        'OC_PUSH_EVERY': str(int(continual.get('worker_push_every', 16))),
        'OLYMPUS_CRL_EPOCH': str(int(time.time())),
        'OLYMPUS_TRACE_DIR': os.path.abspath(trace_dir),
        'PYTHONPATH': pythonpath,
    })


def _terminate_group(proc, grace_s=30.0):
    if proc is None or proc.poll() is not None:
        return
    os.killpg(proc.pid, signal.SIGTERM)
    try:
        proc.wait(timeout=grace_s)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint')
    parser.add_argument('--state-dir')
    parser.add_argument('--trace-dir', default='/tmp/olympus-crl-traces')
    args = parser.parse_args()

    source_config = os.path.abspath(args.config)
    with open(source_config) as handle:
        source_cfg = yaml.safe_load(handle) or {}
    checkpoint = os.path.abspath(_checkpoint_path(source_cfg, args.checkpoint))
    if not checkpoint or not os.path.isfile(checkpoint):
        raise SystemExit(f'checkpoint not found: {checkpoint or "<unset>"}')

    deployment = source_cfg.get('deployment') or {}
    requested_state_dir = (args.state_dir
                           or ((deployment.get('continual') or {}).get('state_dir'))
                           or str(_ROOT / 'olympus' / 'models' / 'crl_sac_service'))
    try:
        cfg, resolved, live_checkpoint = prepare_service_config(
            source_cfg, checkpoint, requested_state_dir)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    lock_path = os.path.abspath(deployment.get(
        'lock_path', '/tmp/olympus-deployment.lock'))
    service_lock = _acquire_service_lock(lock_path)
    learner = listener = None
    old_sigterm = signal.getsignal(signal.SIGTERM)
    old_sigint = signal.getsignal(signal.SIGINT)

    def _shutdown_signal(_signum, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _shutdown_signal)
    signal.signal(signal.SIGINT, _shutdown_signal)
    try:
        learner, manager_addr, manager_key = _start_learner(
            cfg, resolved, _ROOT)
        discovery = cfg.get('deployment', {}).get('discovery', {}) or {}
        listener_bin = os.path.abspath(discovery.get(
            'listener', str(_ROOT / 'astraea_listener')))
        if not os.path.isfile(listener_bin):
            raise SystemExit(f'listener not found: {listener_bin}')
        worker = str(_ROOT / 'olympus' / 'deployment_crl' / 'worker.py')
        env = _listener_environment(
            cfg, resolved, live_checkpoint, manager_addr, manager_key,
            args.trace_dir)
        command = [
            listener_bin,
            '--mode', str(discovery.get('mode', 'mininet')),
            '--cc-name', str(discovery.get('cc_algorithm', 'astraea')),
            '--script', worker,
            '--config', resolved,
            '--model', live_checkpoint,
            '--scan-ms', str(discovery.get('scan_ms', 10)),
            '--ipv4-only', '1' if discovery.get('ipv4_only', True) else '0',
        ]
        listener = subprocess.Popen(
            command, env=env, cwd=str(_ROOT), start_new_session=True)
        print('[crl-service] online SAC learning enabled '
              f'exploration={int(bool(cfg["continual"].get("exploration")))} '
              f'checkpoint={live_checkpoint} replay='
              f'{cfg["continual"]["replay_path"]}', flush=True)

        while True:
            listener_code = listener.poll()
            learner_code = learner.poll()
            if listener_code is not None:
                return listener_code
            if learner_code is not None:
                print(f'[crl-service] learner exited unexpectedly: '
                      f'{learner_code}', flush=True)
                return learner_code or 1
            time.sleep(0.5)
    except KeyboardInterrupt:
        print('\n[crl-service] shutdown requested', flush=True)
        return 0
    finally:
        # Stop socket discovery first, then give the learner time to atomically
        # persist its final model, optimizer state, reward stats, and replay.
        _terminate_group(listener, grace_s=5.0)
        _terminate_group(learner, grace_s=30.0)
        signal.signal(signal.SIGTERM, old_sigterm)
        signal.signal(signal.SIGINT, old_sigint)
        service_lock.close()


if __name__ == '__main__':
    raise SystemExit(main())

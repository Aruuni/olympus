"""
orchestrator.py — parallel episode manager for olympus.

One orchestrator drives both modes; the selected algorithm decides which:
  * single-agent algorithms (MULTI_AGENT = False) run the chosen `environment`,
    and any environment with >1 flow becomes lagged self-play automatically.
  * multi-agent algorithms (MULTI_AGENT = True: mat, ma_dreamer) train every
    flow jointly using the `sweep` block.

The learner, reward, state, and action are hot-swappable through config.yaml:
  runtime.algorithm: td3
  runtime.reward:    tempest
  runtime.action:    cwnd_multiplier

Usage:
  sudo -E env PATH="$PATH" HOME="$HOME" \\
    python olympus/orchestrator.py \\
      --config olympus/config.yaml
"""

import argparse
import csv
import glob
import itertools
import json
import multiprocessing
import os
import random
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback

import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_ENV_DIR = os.path.join(_HERE, 'environments')
# Environment backend type → subfolder under environments/. The link/traffic
# emulator currently provided is Mininet; selecting it picks environments/mininet/.
_DEFAULT_ENV_TYPE = 'mininet'
sys.path.insert(0, _ROOT)
sys.path.insert(0, _HERE)

from olympus.environments import make_env
from olympus.plots.episode_plot import (
    plot as _plot_episode,
    episode_return as _episode_return,
)
from olympus.plots.multi_flow_episode_plot import (
    plot as _plot_multi_episode,
    episode_return as _multi_episode_return,
)
from olympus.common.checkpoint_config import (
    apply_model_config_from_checkpoint,
)
from olympus.common.action_plugins import action_env
from olympus.common.registry import (
    action_module,
    is_multi_agent,
    learner_script,
    reward_module,
    worker_script as resolve_worker_script,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _terminate(proc):
    if proc is None or proc.poll() is not None:
        return
    try:
        pgid = os.getpgid(proc.pid)
    except ProcessLookupError:
        return
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(pgid, sig)
        except ProcessLookupError:
            break
        try:
            proc.wait(timeout=3)
            break
        except subprocess.TimeoutExpired:
            pass


def _wait_or_terminate(proc, timeout=5):
    if proc is None or proc.poll() is not None:
        return
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        _terminate(proc)


def _run_quiet(cmd, timeout=10):
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       timeout=timeout, check=False)
    except Exception:
        pass


def _pkill(pattern: str):
    _run_quiet(['pkill', '-KILL', '-f', pattern])


def _pkill_exact(name: str):
    _run_quiet(['pkill', '-KILL', '-x', name])


def _as_bool(value, default=False):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in ('1', 'true', 'yes', 'y', 'on')
    return default


def _should_plot_episode(outputs, episode):
    if not _as_bool(outputs.get('plot_episodes'), default=True):
        return False
    every_n = int(outputs.get('plot_every_n', 1) or 1)
    if every_n <= 1:
        return True
    return int(episode) % every_n == 0


def _final_runtime_cleanup(learner_port: int):
    """Parent-side cleanup sweep. Preserves checkpoints and run data."""
    print('[orch] runtime cleanup sweep', flush=True)

    # Workers / learners launched by this scaffold. The learner process is also
    # terminated directly, but keep these patterns for orphaned grandchildren.
    _pkill('olympus/algorithms/.*/worker.py')
    _pkill('olympus/algorithms/.*/learner.py')

    # Listener binaries and traffic helpers can survive if a slot process is
    # terminated before run_episode() reaches its own finally block.
    _pkill('astraea_listener')
    _pkill('oc_bridge')
    _pkill_exact('iperf3')
    _pkill_exact('iperf')
    _pkill_exact('ss')
    _pkill_exact('mnexec')

    _run_quiet(['fuser', '-k', f'{int(learner_port)}/tcp'])
    _run_quiet(['mn', '-c'], timeout=20)


def _restore_sudo_user_ownership(path: str) -> None:
    """Give sudo-created run outputs back to the invoking user."""
    uid_raw = os.environ.get('SUDO_UID')
    gid_raw = os.environ.get('SUDO_GID')
    if not uid_raw or not gid_raw or not path or not os.path.exists(path):
        return
    try:
        uid = int(uid_raw)
        gid = int(gid_raw)
    except ValueError:
        return

    for root, dirs, files in os.walk(path):
        for name in dirs + files:
            item = os.path.join(root, name)
            try:
                os.chown(item, uid, gid)
            except OSError:
                pass
    try:
        os.chown(path, uid, gid)
    except OSError:
        pass


def _run_reset_script() -> bool:
    """Run the non-destructive reset script without killing this orchestrator."""
    reset_script = os.path.join(_HERE, 'reset.sh')
    if not os.path.exists(reset_script):
        print(f'[orch] reset script missing: {reset_script}', flush=True)
        return False
    env = dict(os.environ)
    env['SAO_RESET_SKIP_ORCHESTRATOR'] = '1'
    try:
        print(f'[orch] running reset script: {reset_script}', flush=True)
        proc = subprocess.run([reset_script], env=env, check=False)
        if proc.returncode != 0:
            print(f'[orch] reset script exited with code {proc.returncode}', flush=True)
            return False
        return True
    except Exception as e:
        print(f'[orch] reset script failed: {e}', flush=True)
        return False


def _run_link_schedule(env, schedule, episode_start, stop):
    for entry in schedule:
        t_target = episode_start + float(entry['t'])
        while not stop.is_set():
            rem = t_target - time.monotonic()
            if rem <= 0:
                break
            stop.wait(timeout=min(rem, 0.05))
        if stop.is_set():
            return
        try:
            env.set_link(
                bw    = entry.get('bw',    None),
                delay = entry.get('delay', None),
                loss  = entry.get('loss',  None),
            )
            print(f'[orch] link change t={time.monotonic()-episode_start:.1f}s '
                  f'bw={entry.get("bw")} delay={entry.get("delay")}', flush=True)
        except Exception as e:
            print(f'[orch] link change failed: {e}', flush=True)


def _decayed_noise(cfg: dict, episode: int) -> float:
    """
    Linear decay of exploration σ (pre-tanh Gaussian) from noise_start →
    noise_end across noise_decay_episodes episodes.  Per-episode granularity
    is sufficient because one episode ≈ 6000 control steps.
    """
    a_cfg = cfg.get('agent', {}) or {}
    n0 = float(a_cfg.get('noise_start',           0.3))
    n1 = float(a_cfg.get('noise_end',             0.05))
    ne = max(1, int(a_cfg.get('noise_decay_episodes', 300)))
    frac = min(1.0, max(0.0, float(episode) / float(ne)))
    return n0 + (n1 - n0) * frac


def _slug(text: str) -> str:
    text = re.sub(r'[^A-Za-z0-9_.-]+', '_', str(text).strip())
    return text.strip('_') or 'learner'


def _episode_checkpoint_path(checkpoint_path: str, completed_episodes: int) -> str:
    """Return e.g. mbpo_td3_200_cwnd_model.pt for the rolling checkpoint path."""
    directory = os.path.dirname(os.path.abspath(checkpoint_path))
    name = os.path.basename(checkpoint_path)
    stem, ext = os.path.splitext(name)
    ext = ext or '.pt'
    marker = '_cwnd_model'
    if stem.endswith(marker):
        prefix = stem[:-len(marker)]
        filename = f'{prefix}_{int(completed_episodes)}{marker}{ext}'
    else:
        filename = f'{stem}_{int(completed_episodes)}{ext}'
    return os.path.join(directory, filename)


def _lagged_checkpoint_path(checkpoint_path: str, episode: int,
                            lag_episodes: int) -> str:
    """Return the newest immutable checkpoint no newer than episode-lag."""
    lag_episodes = max(0, int(lag_episodes or 0))
    if lag_episodes <= 0:
        return os.path.abspath(checkpoint_path)
    target = max(0, int(episode) - lag_episodes)
    if target <= 0:
        return ''

    exact = _episode_checkpoint_path(checkpoint_path, target)
    if os.path.exists(exact):
        return exact

    directory = os.path.dirname(os.path.abspath(checkpoint_path))
    name = os.path.basename(checkpoint_path)
    stem, ext = os.path.splitext(name)
    ext = ext or '.pt'
    marker = '_cwnd_model'
    if stem.endswith(marker):
        prefix = stem[:-len(marker)]
        suffix = marker
    else:
        prefix = stem
        suffix = ''

    pattern = re.compile(
        r'^{}_(\d+){}{}$'.format(
            re.escape(prefix), re.escape(suffix), re.escape(ext)))
    best_ep = -1
    best_path = ''
    for path in glob.glob(os.path.join(directory, f'{prefix}_*{suffix}{ext}')):
        match = pattern.match(os.path.basename(path))
        if not match:
            continue
        snap_ep = int(match.group(1))
        if snap_ep <= target and snap_ep > best_ep:
            best_ep = snap_ep
            best_path = path
    return best_path


def _as_list(value, default=None) -> list:
    if value is None:
        return list(default or [])
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _episode_rng_seed(ecfg: dict, episode: int) -> int:
    bw = int(round(float(ecfg.get('bw', 0.0)) * 1000.0))
    delay = int(round(float(ecfg.get('delay', 0.0)) * 1000.0))
    return int(ecfg.get('seed', 0) or 0) + int(episode) * 1000003 + bw * 31 + delay


def _materialize_lagged_policy_episode(ecfg: dict, episode: int) -> dict:
    """Sample delayed background flows for lagged-policy fairness episodes."""
    lag_cfg = ecfg.get('lagged_policy') or ecfg.get('lagged_policy_join') or {}
    if not isinstance(lag_cfg, dict) or not _as_bool(lag_cfg.get('enabled'), False):
        return ecfg

    out = dict(ecfg)
    duration = max(1.0, float(out.get('duration', 120.0)))
    rng = random.Random(_episode_rng_seed(out, episode))

    extra_choices = _as_list(
        lag_cfg.get('extra_flows',
                    lag_cfg.get('extra_flow_choices',
                                lag_cfg.get('n_extra_flows', [1]))),
        default=[1],
    )
    extra_choices = [max(0, int(float(v))) for v in extra_choices]
    n_extra = int(rng.choice(extra_choices)) if extra_choices else 1

    join_min = float(lag_cfg.get('join_time_min_s', lag_cfg.get('join_min_s', 5.0)))
    join_max = float(lag_cfg.get('join_time_max_s',
                                  lag_cfg.get('join_max_s', max(join_min, duration - 5.0))))
    join_min = max(0.0, min(join_min, duration - 1.0))
    join_max = max(join_min, min(join_max, duration - 1.0))
    join_times = sorted(rng.uniform(join_min, join_max) for _ in range(n_extra))

    lag_choices = _as_list(
        lag_cfg.get('lag_episodes',
                    lag_cfg.get('policy_lag_episodes', 100)),
        default=[100],
    )
    selected_lag = int(float(rng.choice(lag_choices))) if lag_choices else 100

    start_delays = [0.0] + join_times
    flow_durations = [duration] + [max(1.0, duration - t) for t in join_times]
    out['flows'] = 1 + n_extra
    out['start_delays'] = start_delays
    out['flow_durations'] = flow_durations
    out['fair_flow_start_delays'] = list(start_delays)
    out['fair_flow_durations'] = list(flow_durations)
    out['per_flow_state_logs'] = True
    out['unique_cports'] = False
    out['listener_single_flow'] = False
    # oc_bridge flow ids start at 1. Flow 1 is the initial/train flow; the
    # delayed background flows are attached after it and receive ids 2..N.
    out['lagged_policy_flow_ids'] = list(range(2, 2 + n_extra))
    out['lagged_policy_lag_episodes'] = selected_lag
    out['lagged_policy_deterministic'] = _as_bool(
        lag_cfg.get('deterministic', True), True)
    out['lagged_policy_disable_learning'] = _as_bool(
        lag_cfg.get('disable_learning', True), True)
    out['lagged_policy_require_checkpoint'] = _as_bool(
        lag_cfg.get('require_checkpoint', False), False)
    out['lagged_policy_join_times'] = join_times
    out['lagged_policy_extra_flows'] = n_extra
    # Per-flow base-RTT diversity: each flow (train + lagged background flows)
    # propagates at base_delay * U(min, max). Generated from the same seeded rng
    # so the episode stays reproducible; only set when the env opts in.
    per_flow_delays = _per_flow_rtt_delays(
        out['flows'], out.get('delay', 20.0), out, out, rng)
    if per_flow_delays is not None:
        out['per_flow_delays'] = per_flow_delays
    return out


def _resolve_repo_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(_ROOT, path))


def _environment_selection_label(selection) -> str:
    if isinstance(selection, str):
        return selection
    if isinstance(selection, dict):
        return str(selection.get('environment_setup') or selection.get('name')
                   or selection.get('path') or 'config')
    return 'config'


def _environment_path(env_type: str, name: str) -> str:
    """Built-in environment definitions live in environments/<type>/<name>.yaml."""
    return os.path.join(_ENV_DIR, env_type, f'{name}.yaml')


def _env_backend_type(cfg: dict) -> str:
    """Resolve which network backend to build for this run.

    Prefer metadata recorded by _load_environment_definition, but also honor the
    inline ``environment.type`` selection so callers can branch before the
    environment pool has been materialized.
    """
    meta = cfg.get('_environment_meta') or {}
    env_type = meta.get('type')
    if not env_type and isinstance(cfg.get('environment'), dict):
        env_type = cfg['environment'].get('type')
    return str(env_type or _DEFAULT_ENV_TYPE).strip() or _DEFAULT_ENV_TYPE


def _load_environment_definition(cfg: dict):
    selection = cfg.get('environment')
    if not selection:
        return cfg, {'name': 'config', 'type': '', 'path': '', 'source': 'config'}

    if isinstance(selection, str):
        env_type = _DEFAULT_ENV_TYPE
        if selection.endswith(('.yaml', '.yml')) or os.path.sep in selection:
            name = os.path.splitext(os.path.basename(selection))[0]
            path = _resolve_repo_path(selection)
        else:
            name = selection
            path = _environment_path(env_type, name)
    elif isinstance(selection, dict):
        env_type = str(selection.get('type') or _DEFAULT_ENV_TYPE).strip()
        # `environment_setup` is the scenario file name; `name` is kept as a
        # deprecated fallback so older configs keep loading.
        name = str(selection.get('environment_setup')
                   or selection.get('name') or '').strip()
        raw_path = selection.get('path')
        if raw_path:
            path = _resolve_repo_path(str(raw_path))
            if not name:
                name = os.path.splitext(os.path.basename(path))[0]
        elif name:
            path = _environment_path(env_type, name)
        else:
            raise SystemExit(
                '[orch] environment must define either environment_setup or path')
    else:
        raise SystemExit('[orch] environment must be a name, path, or mapping')

    if not os.path.exists(path):
        raise SystemExit(f'[orch] environment file not found: {path}')

    with open(path) as f:
        env_cfg = yaml.safe_load(f) or {}
    if not isinstance(env_cfg, dict):
        raise SystemExit(f'[orch] environment file must contain a mapping: {path}')
    env_cfg = dict(env_cfg)
    env_name = str(env_cfg.get('name') or name or os.path.splitext(os.path.basename(path))[0])
    return env_cfg, {'name': env_name, 'type': env_type, 'path': path, 'source': 'file'}


def _activate_runtime_blocks(cfg: dict) -> None:
    """Materialize selected algorithm/reward blocks into runtime keys."""
    runtime = cfg.setdefault('runtime', {})
    alg_name = runtime.get('algorithm', 'td3')
    reward_name = runtime.get('reward', 'tempest')
    action_name = runtime.get('action', 'cwnd_multiplier')
    runtime.setdefault('state', 'default_orca')
    runtime.setdefault('action', action_name)

    algorithms = cfg.get('algorithms') or {}
    if algorithms:
        if alg_name not in algorithms:
            choices = ', '.join(sorted(algorithms))
            raise SystemExit(f'[orch] unknown runtime.algorithm={alg_name!r}; choices: {choices}')
        selected = algorithms.get(alg_name) or {}
        cfg['training'] = dict(selected.get('training') or {})
        cfg['agent'] = dict(selected.get('agent') or {})

    try:
        reward_module(reward_name)
    except ModuleNotFoundError as exc:
        missing = exc.name or ''
        # Distinguish "no such reward" from "reward exists but its own imports failed".
        if missing == f'olympus.rewards.{reward_name}':
            raise SystemExit(f'[orch] unknown runtime.reward={reward_name!r}') from exc
        raise SystemExit(
            f'[orch] runtime.reward={reward_name!r} failed to import: {exc}'
        ) from exc
    actions = cfg.get('actions') or {}
    if not actions and action_name == 'cwnd_multiplier':
        # Backward compatibility for configs created before action plugins.
        actions = {'cwnd_multiplier': {}}
        cfg['actions'] = actions
    if action_name not in actions:
        choices = ', '.join(sorted(actions))
        raise SystemExit(
            f'[orch] unknown runtime.action={action_name!r}; choices: {choices}')
    selected_action = dict(actions.get(action_name) or {})
    try:
        plugin = action_module(action_name)
        selected_action = plugin.validate_options(selected_action)
    except (ModuleNotFoundError, ValueError) as exc:
        raise SystemExit(
            f'[orch] invalid runtime.action={action_name!r}: {exc}') from exc
    cfg['action'] = selected_action
    # The global ``reward:`` block holds defaults; a selected algorithm may
    # carry its own ``reward:`` sub-block that overrides them (single-agent and
    # multi-agent rewards are tuned differently, so they cannot share values).
    reward = dict(cfg.get('reward') or {})
    if algorithms:
        reward.update(dict((algorithms.get(alg_name) or {}).get('reward') or {}))
    cfg['reward'] = reward


def _prepare_run(cfg: dict, config_path: str) -> str:
    """Create a per-run output folder and rewrite cfg paths into it."""
    runtime = cfg.setdefault('runtime', {})
    outputs = cfg.setdefault('outputs', {})
    training = cfg.setdefault('training', {})

    alg_name = runtime.get('algorithm', 'td3')
    reward_name = runtime.get('reward', 'tempest')
    root = os.path.abspath(outputs.get('root', os.path.join(_HERE, 'data')))
    timestamp = time.strftime('%Y%m%d-%H%M%S')
    run_name = outputs.get('run_name') or f'{_slug(alg_name)}_{timestamp}'
    run_dir = os.path.abspath(os.path.join(root, run_name))

    checkpoints_dir = os.path.join(run_dir, 'checkpoints')
    episodes_dir = os.path.join(run_dir, 'episodes')
    plots_dir = os.path.join(run_dir, 'plots')
    telemetry_dir = os.path.join(run_dir, 'telemetry')
    for path in (checkpoints_dir, episodes_dir, plots_dir, telemetry_dir):
        os.makedirs(path, exist_ok=True)

    outputs.update({
        'run_dir': run_dir,
        'checkpoints_dir': checkpoints_dir,
        'episodes_dir': episodes_dir,
        'plots_dir': plots_dir,
        'telemetry_dir': telemetry_dir,
        # Backward-compatible alias used by older code paths.
        'traces_dir': episodes_dir,
    })
    training['checkpoint'] = os.path.join(checkpoints_dir, f'{_slug(alg_name)}_cwnd_model.pt')
    training['log_path'] = os.path.join(telemetry_dir, 'learner_metrics.csv')

    resume_from = training.get('resume_from')
    if resume_from:
        resume_from = _resolve_repo_path(str(resume_from))
        if not os.path.exists(resume_from):
            raise SystemExit(f'[orch] resume checkpoint not found: {resume_from}')
        training['resume_from'] = resume_from
        shutil.copy2(resume_from, training['checkpoint'])
        resume_env = resume_from + '.envelope.json'
        if os.path.exists(resume_env):
            shutil.copy2(resume_env, training['checkpoint'] + '.envelope.json')

    meta = {
        'run_name': run_name,
        'created_at': time.strftime('%Y-%m-%d %H:%M:%S %z'),
        'algorithm': alg_name,
        'reward': reward_name,
        'state': runtime.get('state', 'default_orca'),
        'action': runtime.get('action', 'cwnd_multiplier'),
        'environment': _environment_selection_label(cfg.get('environment')),
        'source_config': os.path.abspath(config_path),
        'run_dir': run_dir,
        'resume_from': training.get('resume_from', ''),
    }
    with open(os.path.join(telemetry_dir, 'run_meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    resolved_config = os.path.join(telemetry_dir, 'config.resolved.yaml')
    with open(resolved_config, 'w') as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    outputs['resolved_config'] = resolved_config
    return resolved_config


def _build_sweep_pool(sw: dict, env_meta: dict):
    bws       = sw.get('bws',    [sw.get('bw',    100.0)])
    delays    = sw.get('delays', [sw.get('delay',  20.0)])
    schedules = sw.get('link_schedules', [[]])
    base      = {k: v for k, v in sw.items()
                 if k not in ('bws', 'delays', 'link_schedules', 'bw', 'delay')}
    pool = []
    for bw, delay, sched in itertools.product(bws, delays, schedules):
        ep = dict(base, bw=float(bw), delay=float(delay))
        resolved = []
        for entry in sched:
            e = dict(entry)
            if 'bw_frac' in e:
                e['bw'] = round(float(bw) * e.pop('bw_frac'), 3)
            if 'delay_frac' in e:
                e['delay'] = round(float(delay) * e.pop('delay_frac'), 1)
            resolved.append(e)
        ep['link_schedule'] = resolved
        ep['environment'] = env_meta.get('name', 'config')
        ep['environment_path'] = env_meta.get('path', '')
        pool.append(ep)
    return pool


def _bbr_probe_env(cfg, agent_cfg):
    """Build SAO_BBR_PROBE_* env vars from cfg.

    Top-level ``cfg['bbr_probe']`` wins so the probe can be enabled across all
    pipelines (orca, td3, etc.) with one global block. Falls back to per-
    algorithm ``agent.bbr_probe`` if no top-level block is set.
    """
    top = (cfg.get('bbr_probe') or {}) if isinstance(cfg, dict) else {}
    per_alg = (agent_cfg or {}).get('bbr_probe') or {}
    def _get(key, default):
        if key in top:
            return top[key]
        if key in per_alg:
            return per_alg[key]
        return default
    return {
        'SAO_BBR_PROBE_ENABLED': '1' if bool(_get('enabled', False)) else '0',
        'SAO_BBR_PROBE_INTERVAL_S': str(float(_get('interval_s', 5.0))),
        'SAO_BBR_PROBE_DURATION_S': str(float(_get('duration_s', 0.2))),
        'SAO_BBR_PROBE_FACTOR': str(float(_get('factor', 0.5))),
        'SAO_BBR_MIN_RTT_WINDOW_S': str(float(_get('min_rtt_window_s', 10.0))),
    }


def _tempest_kalman_env(state_cfg):
    state_cfg = state_cfg or {}
    tk_cfg = state_cfg.get('tempest_kalman', {}) or {}

    def _get(key, default):
        flat_key = f'tempest_kalman_{key}'
        return tk_cfg.get(key, state_cfg.get(flat_key, default))

    return {
        'SAO_TEMPEST_KALMAN_INIT_US': str(float(_get('init_us', 20_000.0))),
        'SAO_TEMPEST_KALMAN_Q': str(float(_get('q', 5.0e-4))),
        'SAO_TEMPEST_KALMAN_R_DOWN': str(float(_get('r_down', 1.0e-3))),
        'SAO_TEMPEST_KALMAN_R_UP': str(float(_get('r_up', 5.0))),
        'SAO_TEMPEST_KALMAN_JUMP_THRESH': str(float(_get('jump_thresh', 1.5))),
    }


def _build_pool(cfg):
    env_cfg, env_meta = _load_environment_definition(cfg)
    cfg['_environment_meta'] = env_meta
    has_selected_environment = bool(cfg.get('environment'))

    if 'sweep' in env_cfg:
        return (_build_sweep_pool(env_cfg['sweep'], env_meta),
                bool(env_cfg.get('shuffle', True)))

    if 'experiments' in env_cfg:
        envs = []
        for ep in env_cfg.get('experiments') or [{}]:
            item = dict(ep)
            item.setdefault('environment', env_meta.get('name', 'config'))
            item.setdefault('environment_path', env_meta.get('path', ''))
            envs.append(item)
        return envs, bool(env_cfg.get('shuffle', False))

    if has_selected_environment:
        raise SystemExit(
            f'[orch] environment {env_meta.get("name", "")!r} must define '
            'either sweep or experiments')

    # Backward-compatible fallback: older configs kept sweep/experiments at the
    # top level instead of in olympus/environments/<type>/*.yaml.
    if 'sweep' in cfg:
        return _build_sweep_pool(cfg['sweep'], env_meta), True
    envs = cfg.get('experiments', [{}])
    return list(envs), False


# ── Episode runner ────────────────────────────────────────────────────────────

def run_episode(cfg, ecfg, episode, listener_bin, python_bin,
                mgr_addr, mgr_key, instance_id):
    t_cfg    = cfg.get('training', cfg)
    a_cfg    = cfg.get('agent', {})
    r_cfg    = cfg.get('reward', {}) or {}
    s_cfg    = cfg.get('state_options', {}) or {}
    runtime  = cfg.get('runtime', {}) or {}
    outputs  = cfg.get('outputs', {}) or {}
    duration = int(ecfg.get('duration', cfg.get('duration', 30)))
    alg_name = runtime.get('algorithm', 'td3')
    reward_name = runtime.get('reward', 'tempest')
    state_name = runtime.get('state', 'default_orca')
    ckpt = t_cfg['checkpoint']
    backend_type = _env_backend_type(cfg)
    is_raynet = backend_type == 'raynet'
    plot_trim_tail_s = 0.0 if is_raynet else 5.0

    cport         = int(cfg.get('cport_base', 21000)) + instance_id * 100
    link_schedule = ecfg.get('link_schedule', [])
    hide_link_oracle = _as_bool(
        cfg.get('hide_link_oracle_from_worker',
                cfg.get('inference_hide_link_schedule_from_worker', False)),
        default=False,
    )
    worker_link_bw = 100.0 if hide_link_oracle else float(ecfg.get('bw', 100.0))
    worker_base_rtt_us = 20_000.0 if hide_link_oracle else float(ecfg.get('delay', 20.0)) * 1000
    # Heterogeneous-RTT envs: the live training flow is flow 1, which propagates
    # at its own base RTT, so reference its reward/state to per_flow_delays[0]
    # instead of the shared episode delay. (The lagged background flows replay a
    # checkpoint and compute no learning reward, so only flow 1's reference
    # matters on the single-agent path.)
    _per_flow_delays = ecfg.get('per_flow_delays')
    if _per_flow_delays and not hide_link_oracle:
        worker_base_rtt_us = float(_per_flow_delays[0]) * 1000.0
    worker_link_schedule = [] if hide_link_oracle else link_schedule

    # Oracle state plugin: feed the REAL scheduled link via dedicated env
    # vars that other state plugins do not read. Only set when state==oracle
    # so the oracle feature can never leak into a different state plugin's
    # observation, regardless of who reads what env var. Bypasses
    # hide_link_oracle because explicitly choosing state=oracle is itself
    # the user's opt-in to the schedule.
    oracle_state_active = (state_name == 'oracle')
    oracle_link_bw = float(ecfg.get('bw', 100.0))
    oracle_base_rtt_us = float(ecfg.get('delay', 20.0)) * 1000
    oracle_link_schedule = link_schedule if oracle_state_active else []

    plots_dir = os.path.abspath(outputs['plots_dir'])
    episodes_dir = os.path.abspath(outputs['episodes_dir'])
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(episodes_dir, exist_ok=True)
    state_log = os.path.join(episodes_dir, f'{alg_name}_state_ep{episode:06d}.csv')

    n_flows = int(ecfg.get('flows', 1))
    lagged_flow_ids = [
        int(v) for v in _as_list(ecfg.get('lagged_policy_flow_ids'), default=[])
    ]
    lagged_lag_episodes = int(ecfg.get('lagged_policy_lag_episodes', 0) or 0)
    lagged_checkpoint = ''
    if lagged_flow_ids:
        lagged_checkpoint = _lagged_checkpoint_path(
            ckpt, episode, lagged_lag_episodes)
        if not lagged_checkpoint:
            lagged_checkpoint = os.path.abspath(ckpt)
            print(f'[slot={instance_id}] ep={episode} lagged checkpoint '
                  f'episode-{lagged_lag_episodes} unavailable; using rolling '
                  f'checkpoint for background flows', flush=True)

    fair_flow_starts = ecfg.get('fair_flow_start_delays', ecfg.get('start_delays'))
    if fair_flow_starts is None:
        fair_flow_starts = [0.0] * n_flows
    fair_flow_durations = ecfg.get('fair_flow_durations', ecfg.get('flow_durations'))
    if fair_flow_durations is None:
        fair_flow_durations = []

    worker_env = dict(os.environ)
    worker_env.update({
        # oc_bridge reads OC_PYTHON to fork the worker - must be set.
        'OC_PYTHON':        python_bin,
        'SAO_PYTHON':       python_bin,
        'SAO_LISTENER_CC':  str(cfg.get('listener_cc', 'astraea')),
        'OC_LISTENER_CC':   str(cfg.get('listener_cc', 'astraea')),
        'SAO_ALGORITHM':    alg_name,
        'SAO_REWARD':       reward_name,
        'SAO_STATE':        state_name,
        'OC_STATE':         state_name,
        **action_env(cfg),
        'SAO_ORCA_TARGET_MS': str(float(s_cfg.get('orca_target_ms', 50.0))),
        **_tempest_kalman_env(s_cfg),
        'SAO_CHECKPOINT':   os.path.abspath(ckpt),
        'SAO_MANAGER_ADDR': mgr_addr,
        'SAO_MANAGER_KEY':  mgr_key,
        'SAO_LINK_BW':      str(worker_link_bw),
        'SAO_BASE_RTT_US':  str(worker_base_rtt_us),
        'SAO_INTERVAL_MS':  str(float(a_cfg.get('interval_ms', 20))),
        'SAO_CWND_MIN':     str(int(a_cfg.get('cwnd_min',   10))),
        'SAO_CWND_MAX':     str(int(a_cfg.get('cwnd_max', 10000))),
        'SAO_HIDDEN':       str(int(a_cfg.get('hidden', 128))),
        'SAO_HEAD_HIDDEN':  ('' if a_cfg.get('head_hidden') is None
                             else str(int(a_cfg.get('head_hidden')))),
        'SAO_REC_DIM':      str(int(a_cfg.get('rec_dim', 10))),
        'SAO_ORCA_REC_DIM': str(int(a_cfg.get('rec_dim', 10))),
        'SAO_ORCA_USE_NORMALIZER': '1' if a_cfg.get('use_normalizer', False) else '0',
        'SAO_ORCA_SLOW_START_GATE': (
            '1' if a_cfg.get('slow_start_gate', alg_name == 'orca') else '0'),
        # BBR3-style independent cwnd probe — algorithm- and reward-agnostic.
        # Reads top-level cfg['bbr_probe'] first; falls back to per-algorithm
        # agent.bbr_probe so a single global block enables it across pipelines.
        **_bbr_probe_env(cfg, a_cfg),
        # DreamerV3 worker reads these to build matching encoder/RSSM/actor.
        'SAO_DREAMER_HIDDEN':       str(int(a_cfg.get('hidden',         256))),
        'SAO_DREAMER_EMBED':        str(int(a_cfg.get('embed_dim',      128))),
        'SAO_DREAMER_H_DIM':        str(int(a_cfg.get('h_dim',          256))),
        'SAO_DREAMER_GROUPS':       str(int(a_cfg.get('latent_groups',  8))),
        'SAO_DREAMER_CLASSES':      str(int(a_cfg.get('latent_classes', 8))),
        'SAO_DREAMER_REWARD_BINS':  str(int(a_cfg.get('reward_bins',    255))),
        'SAO_DREAMER_ACTOR_LOG_STD_MIN': str(float(t_cfg.get(
            'actor_log_std_min', -2.0))),
        'SAO_DREAMER_ACTOR_LOG_STD_MAX': str(float(t_cfg.get(
            'actor_log_std_max', 0.0))),
        'SAO_DEAD_FLOW_MS': str(int(a_cfg.get('dead_flow_ms', 1000))),
        'SAO_NOISE_STD':    str(float(_decayed_noise(cfg, episode))),
        'SAO_TRACE_LOG':    state_log,
        'SAO_EPISODE':      str(int(episode)),
        'SAO_HIDE_LINK_ORACLE': '1' if hide_link_oracle else '0',
        # Oracle state plugin only: scheduled-link ground truth via dedicated
        # env vars. Empty when state != oracle so the data is structurally
        # unreachable from other state plugins.
        'SAO_ORACLE_BASE_RTT_US':   str(oracle_base_rtt_us) if oracle_state_active else '',
        'SAO_ORACLE_LINK_SCHEDULE': json.dumps(oracle_link_schedule) if oracle_state_active else '',
        # Option-Critic compatibility aliases. These let the option_critic
        # algorithm keep its richer worker/learner contract while the
        # orchestrator remains shared.
        'OC_CHECKPOINT':     os.path.abspath(ckpt),
        'OC_MANAGER_ADDR':   mgr_addr,
        'OC_MANAGER_KEY':    mgr_key,
        'OC_STATE_LOG':      state_log,
        'OC_EPISODE':        str(int(episode)),
        'SAO_TRACE_LOG_SUFFIX_BY_FLOW': '1' if ecfg.get('per_flow_state_logs') else '0',
        'OC_STATE_LOG_SUFFIX_BY_FLOW':  '1' if ecfg.get('per_flow_state_logs') else '0',
        'OC_CWND_MIN':       str(int(a_cfg.get('cwnd_min', 10))),
        'OC_CWND_MAX':       str(int(a_cfg.get('cwnd_max', 10000))),
        'OC_N_OPTIONS':      str(int(a_cfg.get('n_options', 3))),
        'OC_HIDDEN':         str(int(a_cfg.get('hidden', 128))),
        'OC_ARCH':           str(a_cfg.get('arch', 'mlp')),
        'OC_ACTION_INIT':    str(a_cfg.get('action_init', 'small_spread')),
        'OC_ACTION_INIT_SCALE': str(float(a_cfg.get('action_init_scale', 0.05))),
        'OC_ACTION_LOGSIG_MIN': str(float(a_cfg.get('action_logsig_min', -1.0))),
        'OC_Q_VALUE_CLIP':   str(float(a_cfg.get('q_value_clip', 1500.0))),
        'OC_BETA_TEMPERATURE': str(float(a_cfg.get('beta_temperature', 1.0))),
        'OC_BETA_FLOOR':     str(float(a_cfg.get('beta_floor', 0.0))),
        'OC_EPS_START':      str(float(a_cfg.get('epsilon_start', 0.20))),
        'OC_EPS_END':        str(float(a_cfg.get('epsilon_end', 0.03))),
        'OC_EPS_DECAY':      str(float(a_cfg.get('epsilon_decay', 30000))),
        'OC_DWELL_MIN_STEPS': str(int(a_cfg.get('dwell_min_steps', 1))),
        'OC_DWELL_MAX_STEPS': str(int(a_cfg.get('dwell_max_steps', 0))),
        'OC_TERM_THRESHOLD': str(float(a_cfg.get('term_threshold', 0.5))),
        'OC_DEAD_FLOW_MS':   str(int(a_cfg.get('dead_flow_ms', 1000))),
        'OC_PUSH_EVERY':     str(int(t_cfg.get('worker_push_every', 16))),
        # Compatibility aliases for the current Orca reward plugin.
        'OC_LINK_BW':          str(worker_link_bw),
        'OC_BASE_RTT_US':      str(worker_base_rtt_us),
        'OC_INTERVAL_MS':      str(float(a_cfg.get('interval_ms', 20))),
        'OC_LINK_SCHEDULE':    json.dumps(worker_link_schedule),
        'OC_ORACLE_RTT':       '0' if hide_link_oracle else os.environ.get('OC_ORACLE_RTT', '0'),
        'OC_W_SRTT_DRIFT':     str(float(r_cfg.get('w_srtt_drift', 2.0))),
        'OC_SRTT_DRIFT_CAP':   str(float(r_cfg.get('srtt_drift_cap', 4.0))),
        'OC_ORCA_DELAY_MARGIN_COEF': str(float(r_cfg.get('delay_margin_coef', 1.25))),
        'OC_FAIRNESS_METRIC':  str(r_cfg.get('fairness_metric', 'r_fair')),
        'OC_W_R_FAIR':         str(float(r_cfg.get('w_r_fair', 25.0))),
        'OC_R_FAIR_CAP':       str(float(r_cfg.get('r_fair_cap', 1.0))),
        'OC_W_JAIN':           str(float(r_cfg.get('w_jain', 25.0))),
        'OC_FAIR_FLOW_START_DELAYS': json.dumps(fair_flow_starts),
        'OC_FAIR_FLOW_DURATIONS': json.dumps(fair_flow_durations),
        'SAO_LAGGED_POLICY_CHECKPOINT': lagged_checkpoint,
        'SAO_LAGGED_POLICY_FLOW_IDS': ','.join(str(v) for v in lagged_flow_ids),
        'SAO_LAGGED_POLICY_DETERMINISTIC': (
            '1' if _as_bool(ecfg.get('lagged_policy_deterministic', True), True) else '0'),
        'SAO_LAGGED_POLICY_DISABLE_LEARNING': (
            '1' if _as_bool(ecfg.get('lagged_policy_disable_learning', True), True) else '0'),
        'SAO_LAGGED_POLICY_REQUIRE_CHECKPOINT': (
            '1' if _as_bool(ecfg.get('lagged_policy_require_checkpoint', False), False) else '0'),
    })

    bw_str    = f"{ecfg.get('bw', 100):.0f}"
    delay_str = f"{ecfg.get('delay', 20):.0f}"
    env_name  = str(ecfg.get('environment', ''))
    print(f'[slot={instance_id}] ep={episode}  '
          f'env={env_name or "config"}  bw={bw_str}Mbps  '
          f'delay={delay_str}ms  duration={duration}s', flush=True)

    env_kwargs = {
        'n': n_flows,
        'bw': ecfg.get('bw', 100.0),
        'delay': ecfg.get('delay', 20.0),
        'bdp_mult': ecfg.get('bdp_mult', 4.0),
        'duration': duration,
        'cport': cport,
        'cc_algo': 'astraea',
        'instance_id': instance_id,
        'unique_cports': bool(ecfg.get('unique_cports', False)),
        'per_flow_delays': ecfg.get('per_flow_delays'),
    }
    if is_raynet:
        env_kwargs['environment_config'] = ecfg
    env = make_env(
        backend_type,
        **env_kwargs,
    )
    env.start()

    episode_start = time.monotonic()
    worker_env['SAO_EPISODE_START'] = str(episode_start)
    worker_env['OC_EPISODE_START']  = str(episode_start)  # reward.py reads this
    # Only present when state == oracle so the oracle plugin's schedule
    # parser cannot align timestamps for any other state's process.
    if oracle_state_active:
        worker_env['SAO_ORACLE_EPISODE_START'] = str(episode_start)
    if is_raynet:
        worker_env.update({
            'OC_FLOW_BACKEND': 'raynet',
            'OC_RAYNET_FLOW_ADDR': env.flow_addr,
            'OC_RAYNET_FLOW_KEY': env.flow_key,
        })

    worker_script = resolve_worker_script(alg_name)
    if is_raynet:
        worker_env['OC_FLOW_ID'] = '0'
        worker_env['OC_FLOW_FD'] = '0'
        worker_env['OC_CPORT'] = str(cport)
        listener_cmd = [python_bin, worker_script]
    else:
        listener_cmd = [
            listener_bin, '--cport', str(cport), '--worker', worker_script,
            '--mode', 'mininet', '--scan-ms', str(cfg.get('scan_ms', 20)),
        ]
        listener_single_flow = _as_bool(
            ecfg.get('listener_single_flow', cfg.get('listener_single_flow', True)),
            default=True,
        )
        if listener_single_flow:
            listener_cmd += ['--single-flow', '1']
        if not _as_bool(cfg.get('listener_state_pipe'), default=False):
            listener_cmd += ['--no-state-pipe', '1']

    listener_proc = subprocess.Popen(
        listener_cmd,
        env=worker_env, start_new_session=True)

    sched_stop   = threading.Event()
    sched_thread = None

    try:
        env.run_iperf(
            monitor_interval=float(ecfg.get('measure_interval_s', cfg.get('measure_interval_s', 0.1))),
            start_delays=ecfg.get('start_delays'),
            flow_durations=ecfg.get('flow_durations'),
        )
        if link_schedule and not is_raynet:
            sched_thread = threading.Thread(
                target=_run_link_schedule,
                args=(env, link_schedule, episode_start, sched_stop),
                daemon=True)
            sched_thread.start()
        if not is_raynet:
            time.sleep(duration + 3)
    finally:
        sched_stop.set()
        if sched_thread:
            sched_thread.join(timeout=2)
        if is_raynet:
            env.stop()
            _wait_or_terminate(listener_proc)
        else:
            _terminate(listener_proc)
            env.stop()
        time.sleep(1)

    ep_return = None
    plot_episodes = _should_plot_episode(outputs, episode)
    root, ext = os.path.splitext(state_log)
    ext = ext or '.csv'

    # Per-agent traces: parameter-shared multi-agent models (and single-agent
    # models, which still use the per-agent trace writer) write one trace per
    # agent as `<root>_aK.csv` rather than to `state_log` directly. Render these
    # with the combined multi-flow plot, which names the PDF `ep..._n{n}.pdf`
    # — the path the benchmarks record as each run's individual plot.
    agent_logs = sorted(
        glob.glob(f'{root}_a*{ext}'),
        key=lambda p: (int(re.search(r'_a(\d+)', p).group(1))
                       if re.search(r'_a(\d+)', p) else 10 ** 9),
    )
    if agent_logs:
        n_agents = len(agent_logs)
        ep_return = _multi_episode_return(
            state_log, n_agents=n_agents, trim_tail_s=plot_trim_tail_s)
        if plot_episodes:
            try:
                pdf_path = os.path.join(
                    plots_dir,
                    f'ep{episode:06d}_bw{bw_str}_d{delay_str}_n{n_agents}.pdf')
                plotted_return = _plot_multi_episode(
                    state_log_path=state_log,
                    output=pdf_path,
                    bw=float(ecfg.get('bw', 100.0)),
                    delay=float(ecfg.get('delay', 20.0)),
                    title=(f'Episode {episode}  flows={n_agents}  '
                           f'bw={bw_str}Mbps  delay={delay_str}ms  '
                           f'env={env_name or "config"}  '
                           f'({"scheduled" if link_schedule else "static"})'),
                    link_schedule=link_schedule,
                    n_agents=n_agents,
                    trim_tail_s=plot_trim_tail_s,
                )
                if plotted_return is not None:
                    ep_return = plotted_return
                print(f'[ep_plot] ep={episode} -> {pdf_path}  '
                      f'return={ep_return}', flush=True)
            except Exception as e:
                print(f'[slot={instance_id}] ep={episode} multi-flow plot '
                      f'failed: {e}', flush=True)
        return ep_return, ecfg, link_schedule

    state_logs = []
    if os.path.exists(state_log):
        state_logs.append(state_log)
    if ecfg.get('per_flow_state_logs'):
        flow_logs = sorted(
            glob.glob(f'{root}_flow*{ext}'),
            key=lambda p: int(re.search(r'_flow(\d+)', p).group(1))
            if re.search(r'_flow(\d+)', p) else 10**9,
        )
        if flow_logs:
            state_logs = flow_logs

    if state_logs:
        primary_log = next(
            (p for p in state_logs if re.search(r'_flow0(?:\.|$)', p)),
            state_logs[0],
        )
        try:
            for log_path in state_logs:
                flow_match = re.search(r'_flow(\d+)', log_path)
                flow_suffix = f'_flow{flow_match.group(1)}' if flow_match else ''
                if plot_episodes:
                    pdf_path = os.path.join(
                        plots_dir,
                        f'ep{episode:06d}_bw{bw_str}_d{delay_str}{flow_suffix}.pdf')
                    ret = _plot_episode(
                        state_log_path = log_path,
                        output         = pdf_path,
                        bw             = float(ecfg.get('bw',   100.0)),
                        delay          = float(ecfg.get('delay', 20.0)),
                        title          = (f'Episode {episode}{flow_suffix}  '
                                          f'bw={bw_str}Mbps  delay={delay_str}ms  '
                                          f'env={env_name or "config"}  '
                                          f'({"scheduled" if link_schedule else "static"})'),
                        link_schedule  = link_schedule,
                        trim_tail_s    = plot_trim_tail_s,
                    )
                    if log_path == primary_log:
                        ep_return = ret
                    print(f'[ep_plot] ep={episode}{flow_suffix} -> {pdf_path}  '
                          f'return={ret}', flush=True)
                elif log_path == primary_log:
                    ep_return = _episode_return(
                        log_path, trim_tail_s=plot_trim_tail_s)
        except Exception as e:
            print(f'[slot={instance_id}] ep={episode} plot failed: {e}', flush=True)

    return ep_return, ecfg, link_schedule


# ── Multi-agent (MARL) path ───────────────────────────────────────────────────
#
# Used when the selected algorithm declares MULTI_AGENT = True (mat, ma_dreamer).
# One worker (and one oc_bridge) is launched per flow, each carrying its agent
# index, and the learner trains every flow jointly. Single-agent algorithms in
# multi-flow environments do NOT come through here — they run as lagged
# self-play via run_episode() above.


def _episode_type_label(change_count: int) -> str:
    n = int(change_count)
    return f'{n}_change' if n == 1 else f'{n}_changes'


def _available_tcp_cc() -> set:
    try:
        with open('/proc/sys/net/ipv4/tcp_available_congestion_control') as f:
            return set(f.read().split())
    except OSError:
        return set()


def _ensure_tcp_cc(cc_name: str) -> None:
    cc_name = str(cc_name).strip()
    available = _available_tcp_cc()
    if cc_name in available:
        return

    load_error = ''
    if cc_name == 'astraea' and os.geteuid() == 0:
        module_path = os.path.join(
            _ROOT, 'kernel', 'tcp-astraea', 'tcp_astraea.ko')
        if os.path.exists(module_path):
            result = subprocess.run(
                ['insmod', module_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
            load_error = result.stdout.strip()
            available = _available_tcp_cc()
            if cc_name in available:
                print(
                    f'[orch] loaded TCP congestion control {cc_name!r} '
                    f'from {module_path}',
                    flush=True,
                )
                return

    choices = ' '.join(sorted(available)) or '(unreadable)'
    detail = f' insmod said: {load_error}' if load_error else ''
    raise SystemExit(
        f"[orch] TCP congestion control {cc_name!r} is unavailable "
        f'(available: {choices}). Run `sudo bash setup.sh` before starting '
        f'the experiment.{detail}'
    )


def _sample_range(spec, step=None, rng=None, digits=3) -> float:
    if spec is None:
        raise ValueError('range spec is required')
    vals = list(spec)
    if len(vals) != 2:
        raise ValueError(f'range must have exactly two values, got: {spec!r}')
    lo, hi = sorted((float(vals[0]), float(vals[1])))
    rng = rng or random
    if step is not None:
        step = float(step)
        if step <= 0:
            raise ValueError(f'range step must be > 0, got: {step!r}')
        n = int(round((hi - lo) / step))
        choices = [lo + i * step for i in range(n + 1)]
        choices = [min(max(v, lo), hi) for v in choices]
        return float(rng.choice(choices))
    return round(float(rng.uniform(lo, hi)), int(digits))


def materialize_episode_config(ecfg: dict, rng=None) -> dict:
    """Resolve sampled base BW/RTT and schedule fractions for one MARL episode."""
    rng = rng or random
    out = {k: v for k, v in dict(ecfg).items()
           if not k.startswith('_sample_') and k != '_link_schedule_template'}

    if '_sample_bw_range' in ecfg:
        out['bw'] = _sample_range(
            ecfg['_sample_bw_range'],
            step=ecfg.get('_sample_bw_step'),
            rng=rng,
            digits=int(ecfg.get('_sample_round_digits', 3)),
        )
    else:
        out['bw'] = float(out.get('bw', 100.0))

    if '_sample_delay_range' in ecfg:
        out['delay'] = _sample_range(
            ecfg['_sample_delay_range'],
            step=ecfg.get('_sample_delay_step'),
            rng=rng,
            digits=int(ecfg.get('_sample_round_digits', 3)),
        )
    else:
        out['delay'] = float(out.get('delay', 20.0))

    sched = ecfg.get('_link_schedule_template', ecfg.get('link_schedule', [])) or []
    resolved = []
    for entry in sched:
        e = dict(entry)
        if 'bw_frac' in e:
            e['bw'] = round(float(out['bw']) * e.pop('bw_frac'), 3)
        if 'delay_frac' in e:
            e['delay'] = round(float(out['delay']) * e.pop('delay_frac'), 3)
        resolved.append(e)
    out['link_schedule'] = resolved
    return out


def _parsed_flow_values(spec, default=2):
    if spec is None:
        spec = default
    if isinstance(spec, str):
        text = spec.strip()
        if '-' in text:
            lo, hi = text.split('-', 1)
            vals = range(int(lo), int(hi) + 1)
        else:
            vals = [int(text)]
    elif isinstance(spec, (list, tuple)):
        vals = spec
    else:
        vals = [spec]

    out = []
    for value in vals:
        n = max(1, int(value))
        if n not in out:
            out.append(n)
    return out or [max(1, int(default))]


def _flow_values(spec, default=2):
    """Return flow counts supported by the four-agent training model."""
    out = []
    for value in _parsed_flow_values(spec, default=default):
        n = min(value, 4)
        if n not in out:
            out.append(n)
    return out


def _flow_value(spec, default=2):
    vals = _flow_values(spec, default=default)
    return max(vals)


def _execution_flow_value(spec, default=2):
    """Return the requested count for decentralized evaluation episodes."""
    return max(_parsed_flow_values(spec, default=default))


def _flow_timings(n_flows: int, duration: float, ecfg: dict,
                  sweep_cfg: dict, rng) -> tuple:
    """Resolve per-flow arrival delays and lifetimes for one MARL episode."""
    join_pattern = str(ecfg.get(
        'agent_join_pattern',
        sweep_cfg.get('agent_join_pattern', 'simul'),
    )).lower()
    max_join_frac = float(ecfg.get(
        'agent_join_max_frac',
        sweep_cfg.get('agent_join_max_frac', 0.2),
    ))
    max_join_frac = min(max(max_join_frac, 0.0), 1.0)

    explicit_start_delays = ecfg.get('start_delays')
    if explicit_start_delays is not None:
        start_delays = [
            max(0.0, float(value))
            for value in list(explicit_start_delays)[:n_flows]
        ]
        start_delays.extend([0.0] * (n_flows - len(start_delays)))
    else:
        start_delays = [0.0] * n_flows
        if n_flows > 1 and join_pattern == 'half_late':
            first_half = (n_flows + 1) // 2
            late_t = float(duration) / 2.0
            for index in range(first_half, n_flows):
                start_delays[index] = late_t
        elif n_flows > 1 and join_pattern == 'random':
            cap = max_join_frac * float(duration)
            for index in range(1, n_flows):
                start_delays[index] = rng.uniform(0.0, cap)

    explicit_flow_durations = ecfg.get('flow_durations')
    if explicit_flow_durations is not None:
        flow_durations = [
            max(1.0, float(value))
            for value in list(explicit_flow_durations)[:n_flows]
        ]
        flow_durations.extend(
            max(1.0, float(duration) - start_delays[index])
            for index in range(len(flow_durations), n_flows)
        )
        return start_delays, flow_durations

    lifetime_pattern = str(ecfg.get(
        'flow_lifetime_pattern',
        sweep_cfg.get('flow_lifetime_pattern', 'episode'),
    )).lower()
    available = [
        max(1.0, float(duration) - delay) for delay in start_delays
    ]
    if lifetime_pattern != 'random':
        return start_delays, available

    min_frac = float(ecfg.get(
        'flow_lifetime_min_frac',
        sweep_cfg.get('flow_lifetime_min_frac', 0.25),
    ))
    max_frac = float(ecfg.get(
        'flow_lifetime_max_frac',
        sweep_cfg.get('flow_lifetime_max_frac', 0.9),
    ))
    min_frac = min(max(min_frac, 0.0), 1.0)
    max_frac = min(max(max_frac, min_frac), 1.0)
    flow_durations = []
    for window in available:
        low = max(1.0, min(window, min_frac * window))
        high = max(low, min(window, max_frac * window))
        flow_durations.append(rng.uniform(low, high))

    keep_one = _as_bool(ecfg.get(
        'flow_lifetime_keep_one',
        sweep_cfg.get('flow_lifetime_keep_one', True),
    ), default=True)
    if keep_one and flow_durations:
        anchor = rng.randrange(len(flow_durations))
        flow_durations[anchor] = available[anchor]
    return start_delays, flow_durations


def _per_flow_rtt_delays(n_flows: int, base_delay: float, ecfg: dict,
                         sweep_cfg: dict, rng):
    """Per-flow one-way propagation delays for inter-RTT-diversity episodes.

    When the environment sets ``per_flow_rtt_mult: true`` each flow propagates
    at its own base RTT: the one-way delay of flow i is
    ``base_delay * U(min_mult, max_mult)``, sampled once per flow per episode
    (default range 0.20x .. 5.0x). The Mininet backend applies these on each
    sender's access link (see ``per_flow_delays`` in
    environments/mininet/env.py), so every flow shares the one bottleneck but
    propagates at its own RTT. Returns ``None`` when disabled, leaving the
    single shared-delay topology untouched (legacy behaviour).
    """
    enabled = _as_bool(
        ecfg.get('per_flow_rtt_mult', sweep_cfg.get('per_flow_rtt_mult', False)),
        default=False,
    )
    if not enabled or n_flows < 1:
        return None
    lo = float(ecfg.get('per_flow_rtt_mult_min',
                        sweep_cfg.get('per_flow_rtt_mult_min', 0.20)))
    hi = float(ecfg.get('per_flow_rtt_mult_max',
                        sweep_cfg.get('per_flow_rtt_mult_max', 5.0)))
    lo, hi = sorted((max(0.0, lo), max(0.0, hi)))
    base = max(0.0, float(base_delay))
    return [round(base * rng.uniform(lo, hi), 3) for _ in range(n_flows)]


def _estimate_collection_rate(cfg: dict, n_parallel: int) -> tuple:
    sweep = cfg.get('sweep', {}) or {}
    agent = cfg.get('agent', {}) or {}
    train = cfg.get('training', cfg) or {}
    flow_vals = _flow_values(sweep.get('flows', 1), default=1)
    avg_flows = sum(flow_vals) / float(len(flow_vals))
    flows_label = (str(flow_vals[0]) if len(flow_vals) == 1
                   else f'{flow_vals[0]}-{flow_vals[-1]} avg={avg_flows:.1f}')
    interval_ms = max(1e-3, float(agent.get('interval_ms', 20)))
    rollout_steps = max(
        1, int(train.get('rollout_steps', train.get('min_replay', 1))))
    ceiling = float(n_parallel) * avg_flows * (1000.0 / interval_ms)
    min_update_s = float(rollout_steps) / max(ceiling, 1e-6)
    return flows_label, interval_ms, rollout_steps, ceiling, min_update_s


def _build_pool_marl(cfg):
    """Episode pool for MARL runs.

    Honors the selected ``environment:`` file (its ``sweep``/``experiments``),
    falling back to an inline ``cfg['sweep']`` / ``cfg['experiments']`` for older
    configs, so multi-agent and single-agent runs share the same environment
    presets in ``olympus/environments/<type>/``.
    """
    env_cfg, env_meta = _load_environment_definition(cfg)
    cfg['_environment_meta'] = env_meta
    if 'sweep' in env_cfg:
        sw        = env_cfg['sweep']
        bw_range  = sw.get('bw_range')
        delay_range = sw.get('delay_range', sw.get('rtt_range'))
        bws       = [None] if bw_range is not None else sw.get('bws', [sw.get('bw', 100.0)])
        delays    = [None] if delay_range is not None else sw.get('delays', [sw.get('delay', 20.0)])
        schedules = sw.get('link_schedules', [[]])
        flow_vals = _flow_values(sw.get('flows', 2), default=2)
        base      = {k: v for k, v in sw.items()
                     if k not in (
                         'bws', 'delays', 'link_schedules', 'bw', 'delay',
                         'bw_range', 'delay_range', 'rtt_range',
                         'bw_step', 'delay_step', 'rtt_step',
                         'sample_round_digits', 'samples_per_schedule',
                         'flows',
                     )}
        pool = []
        repeats = max(1, int(sw.get('samples_per_schedule', 1)))
        for flows, bw, delay, sched in itertools.product(flow_vals, bws, delays, schedules):
            for _ in range(repeats):
                ep = dict(base)
                ep['flows'] = int(flows)
                if bw_range is not None:
                    ep['_sample_bw_range'] = list(bw_range)
                    if sw.get('bw_step') is not None:
                        ep['_sample_bw_step'] = float(sw.get('bw_step'))
                else:
                    ep['bw'] = float(bw)
                if delay_range is not None:
                    ep['_sample_delay_range'] = list(delay_range)
                    delay_step = sw.get('delay_step', sw.get('rtt_step'))
                    if delay_step is not None:
                        ep['_sample_delay_step'] = float(delay_step)
                else:
                    ep['delay'] = float(delay)
                if sw.get('sample_round_digits') is not None:
                    ep['_sample_round_digits'] = int(sw.get('sample_round_digits'))
                ep['_link_schedule_template'] = list(sched or [])
                ep['environment'] = env_meta.get('name', 'config')
                ep['environment_path'] = env_meta.get('path', '')
                pool.append(materialize_episode_config(ep) if (
                    bw_range is None and delay_range is None) else ep)
        return pool, bool(env_cfg.get('shuffle', True))
    envs = env_cfg.get('experiments', [{}])
    out = []
    for ep in envs:
        item = dict(ep)
        item.setdefault('environment', env_meta.get('name', 'config'))
        item.setdefault('environment_path', env_meta.get('path', ''))
        out.append(materialize_episode_config(item))
    return out, bool(env_cfg.get('shuffle', False))


def run_episode_marl(cfg, ecfg, episode, listener_bin, python_bin,
                     mgr_addr, mgr_key, instance_id):
    t_cfg    = cfg.get('training', cfg)
    a_cfg    = cfg.get('agent',    {}) or {}
    r_cfg    = cfg.get('reward',   {}) or {}
    runtime  = cfg.get('runtime',  {}) or {}
    outputs  = cfg.get('outputs',  {}) or {}
    duration = int(ecfg.get('duration', cfg.get('duration', 30)))
    alg_name = runtime.get('algorithm', 'ma_dreamer')
    reward_name = runtime.get('reward', 'tempest_fairness_ma')
    state_name = runtime.get('state', 'tempest')
    ckpt = t_cfg['checkpoint']
    backend_type = _env_backend_type(cfg)
    is_raynet = backend_type == 'raynet'
    plot_trim_tail_s = 0.0 if is_raynet else 5.0

    n_flows  = _execution_flow_value(
        ecfg.get('flows', cfg.get('sweep', {}).get('flows', 2)))
    cport    = int(cfg.get('cport_base', 24000)) + instance_id * 100
    link_schedule = ecfg.get('link_schedule', [])

    sweep_cfg = cfg.get('sweep', {}) or {}
    timing_rng = random.Random(
        (instance_id * 1_000_003 + episode) ^ int(cport))
    start_delays, flow_durations = _flow_timings(
        n_flows, duration, ecfg, sweep_cfg, timing_rng)
    per_flow_delays = _per_flow_rtt_delays(
        n_flows, ecfg.get('delay', 20.0), ecfg, sweep_cfg, timing_rng)
    if per_flow_delays is None:
        per_flow_delays = ecfg.get('per_flow_delays')

    plots_dir    = os.path.abspath(outputs['plots_dir'])
    episodes_dir = os.path.abspath(outputs['episodes_dir'])
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(episodes_dir, exist_ok=True)
    state_log = os.path.join(episodes_dir, f'{alg_name}_state_ep{episode:06d}.csv')

    worker_env = dict(os.environ)
    worker_env.update({
        'OC_PYTHON':         python_bin,
        'SAO_PYTHON':        python_bin,
        'SAO_LISTENER_CC':   str(cfg.get('listener_cc', 'astraea')),
        'OC_LISTENER_CC':    str(cfg.get('listener_cc', 'astraea')),
        'SAO_ALGORITHM':     alg_name,
        'SAO_REWARD':        reward_name,
        'SAO_STATE':         state_name,
        'OC_STATE':          state_name,
        **action_env(cfg),
        'SAO_CHECKPOINT':    os.path.abspath(ckpt),
        'SAO_MANAGER_ADDR':  mgr_addr,
        'SAO_MANAGER_KEY':   mgr_key,
        'SAO_LINK_BW':       str(float(ecfg.get('bw',   100.0))),
        'SAO_BASE_RTT_US':   str(float(ecfg.get('delay', 20.0)) * 1000),
        'SAO_INTERVAL_MS':   str(float(a_cfg.get('interval_ms', 20))),
        'SAO_CWND_MIN':      str(int(a_cfg.get('cwnd_min',   10))),
        'SAO_CWND_MAX':      str(int(a_cfg.get('cwnd_max', 10000))),
        'SAO_HIDDEN':        str(int(a_cfg.get('hidden', 128))),
        'SAO_DEAD_FLOW_MS':  str(int(a_cfg.get('dead_flow_ms', 1000))),
        'SAO_NOISE_STD':     str(float(_decayed_noise(cfg, episode))),
        'OC_PUSH_EVERY':      str(int(t_cfg.get('worker_push_every', 16))),
        'SAO_TRACE_LOG':     state_log,
        'SAO_EPISODE':       str(int(episode)),
        # Multi-agent specifics
        'SAO_N_AGENTS':      str(n_flows),
        'SAO_INSTANCE_ID':   str(instance_id),
        'SAO_CPORT_BASE':    str(cport),
        'SAO_MAT_LAYERS':    str(int(a_cfg.get('n_layers', 2))),
        'SAO_MAT_HEADS':     str(int(a_cfg.get('n_heads',  4))),
        'SAO_DREAMER_HIDDEN': str(int(a_cfg.get('hidden', 256))),
        'SAO_DREAMER_EMBED': str(int(a_cfg.get('embed_dim', 128))),
        'SAO_DREAMER_H_DIM': str(int(a_cfg.get('h_dim', 256))),
        'SAO_DREAMER_GROUPS': str(int(a_cfg.get('latent_groups', 8))),
        'SAO_DREAMER_CLASSES': str(int(a_cfg.get('latent_classes', 8))),
        'SAO_DREAMER_REWARD_BINS': str(
            int(t_cfg.get('reward_bins', a_cfg.get('reward_bins', 255)))),
        'SAO_DREAMER_ACTOR_LOG_STD_MIN': str(
            float(t_cfg.get('actor_log_std_min', -2.0))),
        'SAO_DREAMER_ACTOR_LOG_STD_MAX': str(
            float(t_cfg.get('actor_log_std_max', 0.0))),
        # Reward env vars used by Tempest and the legacy MAT reward.
        'OC_LINK_BW':          str(float(ecfg.get('bw',   100.0))),
        'OC_BASE_RTT_US':      str(float(ecfg.get('delay', 20.0)) * 1000),
        'OC_INTERVAL_MS':      str(float(a_cfg.get('interval_ms', 20))),
        'OC_LINK_SCHEDULE':    json.dumps(link_schedule),
        'OC_FAIR_FLOW_START_DELAYS': json.dumps(start_delays),
        'OC_FAIR_FLOW_DURATIONS': json.dumps(flow_durations),
    })
    worker_env.update(_bbr_probe_env(cfg, a_cfg))
    worker_env.update(_tempest_kalman_env(cfg.get('state_options', {}) or {}))

    bw_str    = f"{ecfg.get('bw', 100):.0f}"
    delay_str = f"{ecfg.get('delay', 20):.0f}"
    delays_str = ('simul' if max(start_delays) == 0
                  else f'delays={[round(x,1) for x in start_delays]}')
    rtts_str = ('' if not per_flow_delays
                else f'  rtts={[round(x,1) for x in per_flow_delays]}ms')
    print(f'[slot={instance_id}] ep={episode}  flows={n_flows}  '
          f'bw={bw_str}Mbps  delay={delay_str}ms  duration={duration}s  '
          f'{delays_str}  '
          f'lifetimes={[round(x,1) for x in flow_durations]}{rtts_str}', flush=True)

    env_kwargs = {
        'n': n_flows,
        'bw': ecfg.get('bw', 100.0),
        'delay': ecfg.get('delay', 20.0),
        'bdp_mult': ecfg.get('bdp_mult', 4.0),
        'duration': duration,
        'cport': cport,
        'cc_algo': str(cfg.get('listener_cc', 'astraea')),
        'instance_id': instance_id,
        'unique_cports': True,
        'per_flow_delays': per_flow_delays,
    }
    if is_raynet:
        env_kwargs['environment_config'] = ecfg
    env = make_env(backend_type, **env_kwargs)
    env.start()

    episode_start = time.monotonic()
    worker_env['SAO_EPISODE_START'] = str(episode_start)
    worker_env['OC_EPISODE_START']  = str(episode_start)
    if is_raynet:
        worker_env.update({
            'OC_FLOW_BACKEND': 'raynet',
            'OC_RAYNET_FLOW_ADDR': env.flow_addr,
            'OC_RAYNET_FLOW_KEY': env.flow_key,
        })

    worker_script = resolve_worker_script(alg_name)
    listener_procs = []
    for agent_id in range(n_flows):
        listener_cport = cport + agent_id
        listener_env = dict(worker_env)
        listener_env['SAO_AGENT_ID'] = str(agent_id)
        listener_env['OC_FLOW_ID'] = str(agent_id)
        # Heterogeneous-RTT envs: each flow propagates at its own base RTT, so
        # its reward/state reference must be its own delay, not the shared
        # episode base. per_flow_delays[i] is flow i's one-way delay in ms,
        # matching the existing OC_BASE_RTT_US = delay*1000 convention.
        if per_flow_delays and agent_id < len(per_flow_delays):
            flow_rtt_us = str(float(per_flow_delays[agent_id]) * 1000.0)
            listener_env['OC_BASE_RTT_US'] = flow_rtt_us
            listener_env['SAO_BASE_RTT_US'] = flow_rtt_us
        if is_raynet:
            listener_env['OC_FLOW_FD'] = str(agent_id)
            listener_env['OC_CPORT'] = str(listener_cport)
            listener_cmd = [python_bin, worker_script]
        else:
            listener_cmd = [
                listener_bin, '--cport', str(listener_cport), '--worker', worker_script,
                '--mode', 'mininet', '--scan-ms', str(cfg.get('scan_ms', 20)),
            ]
            if _as_bool(cfg.get('listener_single_flow'), default=True):
                listener_cmd += ['--single-flow', '1']
            if not _as_bool(cfg.get('listener_state_pipe'), default=False):
                listener_cmd += ['--no-state-pipe', '1']
        listener_procs.append(subprocess.Popen(
            listener_cmd, env=listener_env, start_new_session=True))

    sched_stop   = threading.Event()
    sched_thread = None
    try:
        env.run_iperf(
            monitor_interval=float(ecfg.get(
                'measure_interval_s', cfg.get('measure_interval_s', 0.1))),
            start_delays=start_delays,
            flow_durations=flow_durations,
        )
        if link_schedule and not is_raynet:
            sched_thread = threading.Thread(
                target=_run_link_schedule,
                args=(env, link_schedule, episode_start, sched_stop),
                daemon=True)
            sched_thread.start()
        if not is_raynet:
            time.sleep(duration + 3)
    finally:
        sched_stop.set()
        if sched_thread:
            sched_thread.join(timeout=2)
        if is_raynet:
            env.stop()
            for listener_proc in listener_procs:
                _wait_or_terminate(listener_proc)
        else:
            for listener_proc in listener_procs:
                _terminate(listener_proc)
            env.stop()
        time.sleep(1)

    ep_return = _multi_episode_return(
        state_log, n_agents=n_flows, trim_tail_s=plot_trim_tail_s)
    plot_episodes = _should_plot_episode(outputs, episode)
    try:
        if plot_episodes:
            pdf_path = os.path.join(plots_dir,
                                    f'ep{episode:06d}_bw{bw_str}_d{delay_str}_n{n_flows}.pdf')
            plotted_return = _plot_multi_episode(
                state_log_path = state_log,
                output         = pdf_path,
                bw             = float(ecfg.get('bw',   100.0)),
                delay          = float(ecfg.get('delay', 20.0)),
                title          = (f'Episode {episode}  flows={n_flows}  '
                                  f'bw={bw_str}Mbps  delay={delay_str}ms  '
                                  f'({"scheduled" if link_schedule else "static"})'),
                link_schedule  = link_schedule,
                n_agents       = n_flows,
                trim_tail_s    = plot_trim_tail_s,
            )
            if plotted_return is not None:
                ep_return = plotted_return
            print(f'[ep_plot] ep={episode} -> {pdf_path}  return={ep_return}', flush=True)
    except Exception as e:
        print(f'[slot={instance_id}] ep={episode} multi-flow plot failed: {e}',
              flush=True)

    if ep_return is not None:
        print(f'[orch] ep={episode} sum-of-agents return={ep_return:.1f}', flush=True)

    return ep_return, ecfg, link_schedule


def run_episode_auto(cfg, ecfg, episode, listener_bin, python_bin,
                     mgr_addr, mgr_key, instance_id):
    """Dispatch one episode to the selected environment backend.

    The selected algorithm owns the single-agent / multi-agent split. Backend
    differences are handled inside the shared episode functions.
    """
    alg_name = (cfg.get('runtime', {}) or {}).get('algorithm', 'td3')
    episode_fn = run_episode_marl if is_multi_agent(alg_name) else run_episode
    return episode_fn(cfg, ecfg, episode, listener_bin, python_bin,
                      mgr_addr, mgr_key, instance_id)


def _materialize_single_episode(ecfg: dict, episode: int) -> dict:
    """Materialize one single-agent episode.

    A single-agent learner in a multi-flow environment runs as lagged
    self-play: flow 1 is the live/training flow and the remaining flows replay a
    lagged checkpoint of the same policy. ``flows`` may be a scalar or a
    list/range/string (e.g. ``[1, 2, 3, 4]``); when it offers >1 flow and the
    environment has no explicit ``lagged_policy`` block, synthesize one whose
    ``extra_flows`` carries every candidate count so the lagged materializer
    samples a per-episode flow count that mirrors ``flows``. If the environment
    uses ``agent_join_pattern: random`` the background flows join over the same
    late-join window (interleaved joins, like the lagged_fairness self-play
    example) instead of all starting at t=0. Single-flow episodes and episodes
    with an explicit lagged_policy block pass straight through to the
    lagged-policy materializer.
    """
    has_lag = isinstance(
        ecfg.get('lagged_policy') or ecfg.get('lagged_policy_join'), dict)
    if not has_lag:
        flow_choices = _parsed_flow_values(ecfg.get('flows', 1), default=1)
        max_flows = max(flow_choices) if flow_choices else 1
        if max_flows > 1:
            ecfg = dict(ecfg)
            duration = float(ecfg.get('duration', 120.0) or 120.0)
            if str(ecfg.get('agent_join_pattern', '')).lower() == 'random':
                join_max = max(
                    0.0, float(ecfg.get('agent_join_max_frac', 0.2)) * duration)
            else:
                join_max = 0.0
            ecfg['lagged_policy'] = {
                'enabled': True,
                # Each candidate flow count minus the live flow; the lagged
                # materializer picks one per episode so the count tracks `flows`.
                'extra_flows': sorted({max(0, n - 1) for n in flow_choices}),
                'join_time_min_s': 0.0,
                'join_time_max_s': join_max,
                'lag_episodes': int(ecfg.get('self_play_lag_episodes', 100)),
            }
        elif not isinstance(ecfg.get('flows', 1), int):
            # Single-flow env that declared `flows` as a list/range/string;
            # collapse to a concrete scalar so downstream int() reads are safe.
            ecfg = dict(ecfg)
            ecfg['flows'] = 1
    return _materialize_lagged_policy_episode(ecfg, episode)


# ── Slot process ──────────────────────────────────────────────────────────────

def _slot_process(instance_id, work_q, result_q,
                  cfg, listener_bin, python_bin, learner_state, multi):
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    while True:
        item = work_q.get()
        if item is None:
            break
        ep, ecfg = item
        episode_wall_start = time.monotonic()
        try:
            ep_return, ecfg_out, link_sched = run_episode_auto(
                cfg, ecfg, ep, listener_bin, python_bin,
                learner_state['mgr_addr'], learner_state['mgr_key'], instance_id)
        except Exception as e:
            print(f'[slot={instance_id}] ep={ep} error: {e}\n{traceback.format_exc()}',
                  flush=True)
            ep_return  = None
            ecfg_out   = ecfg
            link_sched = ecfg.get('link_schedule', [])
        episode_wall_s = time.monotonic() - episode_wall_start
        result_q.put((ep, ecfg_out, ep_return, link_sched, episode_wall_s))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config',   required=True)
    ap.add_argument('--episodes', type=int, default=None)
    ap.add_argument('--environment', default=None,
                    help='environment_setup name or yaml path; e.g. dynamic or static')
    ap.add_argument('--env-type', default=None,
                    help=f'environment backend type (subfolder of environments/); '
                         f'default {_DEFAULT_ENV_TYPE}')
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.environment or args.env_type:
        existing = cfg.get('environment')
        env_type = args.env_type or (
            existing.get('type') if isinstance(existing, dict) else None
        ) or _DEFAULT_ENV_TYPE
        if args.environment:
            if args.environment.endswith(('.yaml', '.yml')) or os.path.sep in args.environment:
                cfg['environment'] = {'type': env_type, 'path': args.environment}
            else:
                cfg['environment'] = {'type': env_type, 'environment_setup': args.environment}
        else:
            # Only the type changed; keep the existing setup selection.
            sel = dict(existing) if isinstance(existing, dict) else {}
            sel['type'] = env_type
            cfg['environment'] = sel
    _activate_runtime_blocks(cfg)
    resume_from = (cfg.get('training', {}) or {}).get('resume_from')
    if resume_from:
        resume_from = _resolve_repo_path(str(resume_from))
        model_config = apply_model_config_from_checkpoint(cfg, resume_from)
        if model_config:
            _activate_runtime_blocks(cfg)
            cfg.setdefault('training', {})['resume_from'] = resume_from
            print(f'[orch] using model config: {model_config}', flush=True)
    # Multi-agent learners size their joint models from sweep.flows, but the
    # sweep lives in the environment yaml, not the training config. Surface it
    # in the resolved config the learner receives so max_agents matches the
    # environment instead of silently falling back to 2.
    if 'sweep' not in cfg:
        env_cfg, _ = _load_environment_definition(cfg)
        env_flows = (env_cfg.get('sweep') or {}).get('flows')
        if env_flows is not None:
            cfg['sweep'] = {'flows': env_flows}
    learner_config_path = _prepare_run(cfg, args.config)

    t_cfg = cfg.get('training', cfg)
    paths = cfg.get('paths', {}) or {}
    backend_type = _env_backend_type(cfg)
    if 'py' not in paths:
        raise SystemExit('[orch] config.yaml must define paths.py')
    if backend_type != 'raynet' and 'listener' not in paths:
        raise SystemExit('[orch] config.yaml must define paths.listener for Mininet backends')
    listener_bin = os.path.abspath(paths.get('listener', '')) if backend_type != 'raynet' else ''
    python_bin   = os.path.abspath(paths['py'])
    n_episodes   = args.episodes or cfg.get('episodes', None)
    n_parallel   = max(1, int(cfg.get('n_parallel', 1)))
    episode_ckpt_every = max(0, int(
        t_cfg.get('checkpoint_every_episodes',
                  t_cfg.get('save_every_episodes', 0)) or 0))

    runtime = cfg.get('runtime', {}) or {}
    alg_name = runtime.get('algorithm', 'td3')
    multi = is_multi_agent(alg_name)
    if backend_type != 'raynet':
        _ensure_tcp_cc(cfg.get('listener_cc', 'astraea'))
    outputs = cfg.get('outputs', {}) or {}
    run_dir = os.path.abspath(outputs['run_dir'])
    plots_dir = os.path.abspath(outputs['plots_dir'])
    episodes_dir = os.path.abspath(outputs['episodes_dir'])
    telemetry_dir = os.path.abspath(outputs['telemetry_dir'])
    checkpoints_dir = os.path.abspath(outputs['checkpoints_dir'])
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(episodes_dir, exist_ok=True)
    os.makedirs(telemetry_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)
    returns_csv = os.path.join(episodes_dir, 'episode_returns.csv')

    print(f'[orch] n_parallel={n_parallel}', flush=True)
    print(f'[orch] run_dir={run_dir}', flush=True)
    if episode_ckpt_every > 0:
        print(f'[orch] episode checkpoint copies every {episode_ckpt_every} '
              'completed episodes', flush=True)

    selected_learner_script = learner_script(alg_name)
    learner_port   = int(cfg.get('learner', {}).get('port', 6301))
    shutdown_requested = {'seen': False}

    def _signal_shutdown(signum, _frame):
        if shutdown_requested['seen']:
            print(f'\n[orch] signal {signum} repeated - forcing exit', flush=True)
            os._exit(128 + int(signum))
        shutdown_requested['seen'] = True
        raise KeyboardInterrupt

    def _start_learner():
        subprocess.run(f'fuser -k {learner_port}/tcp 2>/dev/null; true', shell=True)
        time.sleep(1)

        learner_env = dict(os.environ)
        learner_env['SAO_STATE'] = str(runtime.get('state', 'default_orca'))
        learner_env['OC_STATE'] = str(runtime.get('state', 'default_orca'))
        learner_env.update(action_env(cfg))
        learner_env['SAO_ORCA_TARGET_MS'] = str(float(
            (cfg.get('state_options', {}) or {}).get('orca_target_ms', 50.0)))
        learner_env.update(_tempest_kalman_env(cfg.get('state_options', {}) or {}))
        learner_env['SAO_REC_DIM'] = str(int((cfg.get('agent', {}) or {}).get('rec_dim', 10)))
        learner_env['SAO_ORCA_REC_DIM'] = learner_env['SAO_REC_DIM']
        proc = subprocess.Popen(
            [python_bin, selected_learner_script, '--config', learner_config_path,
             '--port', str(learner_port)],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, env=learner_env, start_new_session=True)
        key = ''
        for line in proc.stdout:
            sys.stdout.write('[learner] ' + line)
            sys.stdout.flush()
            if line.startswith('SAO_MANAGER_KEY='):
                key = line.strip().split('=', 1)[1]
                break
            if proc.poll() is not None:
                for leftover in proc.stdout:
                    sys.stdout.write('[learner] ' + leftover)
                sys.stdout.flush()
                raise RuntimeError(f'learner exited (code={proc.returncode}) before becoming ready')
        if not key:
            raise RuntimeError('learner closed stdout without printing SAO_MANAGER_KEY')
        return proc, key

    learner_proc, mgr_key = _start_learner()

    old_sigterm = signal.getsignal(signal.SIGTERM)
    old_sigint = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGTERM, _signal_shutdown)
    signal.signal(signal.SIGINT, _signal_shutdown)
    interrupted = False

    _mp_mgr       = multiprocessing.Manager()
    learner_state = _mp_mgr.dict({
        'mgr_addr': f'127.0.0.1:{learner_port}',
        'mgr_key':  mgr_key,
    })
    _lproc = {'proc': learner_proc}

    def _fwd(proc):
        for line in proc.stdout:
            sys.stdout.write('[learner] ' + line)
            sys.stdout.flush()
    threading.Thread(target=_fwd, args=(learner_proc,), daemon=True).start()

    stop_watchdog = threading.Event()

    def _learner_watchdog():
        while not stop_watchdog.is_set():
            time.sleep(2)
            if stop_watchdog.is_set():
                break
            proc = _lproc['proc']
            if proc.poll() is not None:
                print(f'[orch] learner died (exit={proc.returncode}) - restarting in 3s',
                      flush=True)
                time.sleep(3)
                try:
                    new_proc, new_key = _start_learner()
                    _lproc['proc'] = new_proc
                    learner_state['mgr_key']  = new_key
                    learner_state['mgr_addr'] = f'127.0.0.1:{learner_port}'
                    threading.Thread(target=_fwd, args=(new_proc,), daemon=True).start()
                    print(f'[orch] learner restarted key={new_key[:8]}...', flush=True)
                except Exception as e:
                    print(f'[orch] learner restart failed: {e}', flush=True)

    threading.Thread(target=_learner_watchdog, daemon=True).start()

    print(f'[orch] learner ready addr={learner_state["mgr_addr"]}', flush=True)
    training_start = time.monotonic()

    if multi:
        pool, do_shuffle = _build_pool_marl(cfg)
        env_meta = {}
        _, interval_ms, rollout_steps, ceiling, min_update_s = (
            _estimate_collection_rate(cfg, n_parallel))
        print(f'[orch] MARL collection ceiling~{ceiling:.0f} transitions/s '
              f'({n_parallel} slots x {1000.0 / interval_ms:.0f}Hz); batch '
              f'threshold={rollout_steps} -> learner updates >= ~{min_update_s:.1f}s apart',
              flush=True)
    else:
        pool, do_shuffle = _build_pool(cfg)
        env_meta = cfg.get('_environment_meta', {}) or {}
    print(f'[orch] mode={"multi-agent" if multi else "single-agent"}  '
          f'environment={env_meta.get("name", "config")}  '
          f'pool={len(pool)}  shuffle={do_shuffle}', flush=True)

    work_q   = multiprocessing.Queue()
    result_q = multiprocessing.Queue()

    slot_procs = []
    for i in range(n_parallel):
        p = multiprocessing.Process(
            target = _slot_process,
            args   = (i, work_q, result_q, cfg,
                      listener_bin, python_bin, learner_state, multi),
            daemon = True,
        )
        p.start()
        slot_procs.append(p)

    pending         = list(pool)
    if do_shuffle:
        random.shuffle(pending)
    episode_counter = 0
    in_flight       = 0

    def _next():
        nonlocal pending, episode_counter
        if n_episodes is not None and episode_counter >= n_episodes:
            return None
        if not pending:
            pending.extend(pool)
            if do_shuffle:
                random.shuffle(pending)
        ep   = episode_counter
        if multi:
            ecfg = materialize_episode_config(pending.pop(0))
        else:
            ecfg = _materialize_single_episode(pending.pop(0), ep)
        episode_counter += 1
        return ep, ecfg

    returns_fields = [
        'episode', 'environment', 'bw', 'delay', 'scheduled',
        'schedule_changes', 'episode_type', 'flows', 'return',
        'elapsed_s', 'episode_wall_s', 'completed_at',
    ]
    pending_episode_ckpts = set()
    missing_episode_ckpt_warned = set()

    def _ensure_returns_csv_header():
        if not os.path.exists(returns_csv):
            with open(returns_csv, 'w', newline='') as f:
                csv.DictWriter(f, fieldnames=returns_fields).writeheader()
            return list(returns_fields)
        try:
            with open(returns_csv, newline='') as f:
                reader = csv.DictReader(f)
                existing = list(reader.fieldnames or [])
                rows = list(reader)
        except (OSError, csv.Error):
            return list(returns_fields)
        if existing == returns_fields:
            return list(returns_fields)
        if all(field in existing for field in returns_fields):
            return existing
        merged = list(existing)
        for field in returns_fields:
            if field not in merged:
                merged.append(field)
        tmp_path = f'{returns_csv}.tmp.{os.getpid()}'
        with open(tmp_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=merged)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        os.replace(tmp_path, returns_csv)
        return merged

    def _log_return(ep, ecfg, ep_return, link_sched, episode_wall_s):
        env_name = ecfg.get('environment') or env_meta.get('name', '')
        schedule_changes = len(link_sched or [])
        episode_type = f'{schedule_changes}_change' if schedule_changes == 1 else f'{schedule_changes}_changes'
        elapsed_s = time.monotonic() - training_start
        active_returns_fields = _ensure_returns_csv_header()
        with open(returns_csv, 'a', newline='') as f:
            w = csv.DictWriter(f, fieldnames=active_returns_fields)
            w.writerow({
                'episode': ep,
                'environment': env_name,
                'bw': ecfg.get('bw', 100),
                'delay': ecfg.get('delay', 20),
                'scheduled': int(bool(link_sched)),
                'schedule_changes': schedule_changes,
                'episode_type': episode_type,
                'flows': ecfg.get('flows', 1),
                'return': f'{ep_return:.4f}' if ep_return is not None else '',
                'elapsed_s': f'{elapsed_s:.3f}',
                'episode_wall_s': f'{float(episode_wall_s):.3f}',
                'completed_at': time.strftime('%Y-%m-%d %H:%M:%S %z'),
            })
        print(f'[orch] ep={ep}  env={env_name}  type={episode_type}  '
              f'return={ep_return}  elapsed={elapsed_s/3600.0:.2f}h',
              flush=True)

        if (ep + 1) % 10 == 0:
            _generate_returns_plot()

    def _generate_returns_plot():
        try:
            from olympus.plots.plot_returns_watcher import generate_plot
            generate_plot(
                csv_path    = returns_csv,
                output      = os.path.join(telemetry_dir, 'episode_returns.pdf'),
                learner_log = t_cfg.get('log_path',
                                        t_cfg['checkpoint'].replace('.pt', '_log.csv')))
        except Exception as e:
            print(f'[orch] returns plot failed: {e}', flush=True)

    def _copy_episode_checkpoint(completed_count: int) -> bool:
        src = os.path.abspath(t_cfg['checkpoint'])
        if not os.path.exists(src):
            if completed_count not in missing_episode_ckpt_warned:
                print(f'[orch] episode checkpoint {completed_count} pending; '
                      f'rolling checkpoint not written yet: {src}', flush=True)
                missing_episode_ckpt_warned.add(completed_count)
            return False
        try:
            size0 = os.path.getsize(src)
            mtime0 = os.path.getmtime(src)
            time.sleep(0.05)
            if os.path.getsize(src) != size0 or os.path.getmtime(src) != mtime0:
                return False
        except OSError:
            return False

        dst = _episode_checkpoint_path(src, completed_count)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        tmp = f'{dst}.tmp.{os.getpid()}'
        try:
            shutil.copy2(src, tmp)
            os.replace(tmp, dst)
            env_src = src + '.envelope.json'
            if os.path.exists(env_src):
                env_dst = dst + '.envelope.json'
                env_tmp = f'{env_dst}.tmp.{os.getpid()}'
                shutil.copy2(env_src, env_tmp)
                os.replace(env_tmp, env_dst)
        finally:
            for path in (tmp, f'{dst}.envelope.json.tmp.{os.getpid()}'):
                try:
                    if os.path.exists(path):
                        os.unlink(path)
                except OSError:
                    pass
        print(f'[orch] saved episode checkpoint completed={completed_count} '
              f'path={dst}', flush=True)
        return True

    def _flush_episode_checkpoints():
        for completed_count in sorted(list(pending_episode_ckpts)):
            if _copy_episode_checkpoint(completed_count):
                pending_episode_ckpts.discard(completed_count)

    def _maybe_save_episode_checkpoint(completed_count: int):
        if episode_ckpt_every <= 0 or completed_count <= 0:
            return
        if completed_count % episode_ckpt_every == 0:
            pending_episode_ckpts.add(completed_count)
        _flush_episode_checkpoints()

    for _ in range(n_parallel):
        item = _next()
        if item is not None:
            work_q.put(item)
            in_flight += 1

    completed_episodes = 0
    try:
        while in_flight > 0:
            ep, ecfg, ep_return, link_sched, episode_wall_s = result_q.get()
            in_flight -= 1
            completed_episodes += 1
            _log_return(ep, ecfg, ep_return, link_sched, episode_wall_s)
            _maybe_save_episode_checkpoint(completed_episodes)

            item = _next()
            if item is not None:
                work_q.put(item)
                in_flight += 1

    except KeyboardInterrupt:
        interrupted = True
        print('\n[orch] interrupted - draining slots', flush=True)

    finally:
        stop_watchdog.set()
        for _ in slot_procs:
            try: work_q.put(None)
            except Exception: pass

        for p in slot_procs:
            p.join(timeout=3 if shutdown_requested['seen'] else 10)
            if p.is_alive():
                p.terminate()
                p.join(timeout=2)
            if p.is_alive():
                p.kill()

        _terminate(_lproc['proc'])
        if interrupted or shutdown_requested['seen']:
            if not _run_reset_script():
                _final_runtime_cleanup(learner_port)
        else:
            _final_runtime_cleanup(learner_port)
        try:
            _mp_mgr.shutdown()
        except Exception:
            pass
        if os.path.exists(returns_csv):
            _generate_returns_plot()
        if pending_episode_ckpts:
            _flush_episode_checkpoints()
            if pending_episode_ckpts:
                pending = ', '.join(str(x) for x in sorted(pending_episode_ckpts))
                print(f'[orch] episode checkpoint copy still pending for '
                      f'completed={pending}', flush=True)
        _restore_sudo_user_ownership(run_dir)
        signal.signal(signal.SIGTERM, old_sigterm)
        signal.signal(signal.SIGINT, old_sigint)

    print('[orch] done', flush=True)


if __name__ == '__main__':
    main()

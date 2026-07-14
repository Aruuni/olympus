#!/usr/bin/env python3
"""Deterministic checkpoint evaluation through the Olympus orchestrator.

Accepts either a legacy training config plus ``--checkpoint`` or a versioned
``kind: olympus-eval`` manifest describing checkpoint × scenario × backend
experiments. Every matrix entry runs sequentially; environments remain parallel
inside each entry.
"""

import argparse
import copy
import os
from pathlib import Path
import queue
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time

import yaml

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from olympus.common.checkpoint_config import load_checkpoint_config
from olympus.common.eval_manifest import (
    is_eval_manifest, load_manifest, scenario_episode_count,
)
from olympus.environments import validate_scenario
from olympus.train import (
    C, _Dashboard, _EPISODE_RE, _fmt_time, _handle_line, _print, _reader,
    _term_width,
)

_LOG_DIR = Path(os.environ.get('OLYMPUS_EVAL_LOG_DIR', str(_HERE / 'logs')))


class _EvalDashboard(_Dashboard):
    def _learner_line(self, width: int) -> str:
        return f'{C.GREY}eval      deterministic checkpoint inference{C.RESET}'

    def render(self, force: bool = False) -> None:
        if self.active:
            return super().render(force=force)
        now = time.monotonic()
        if force or now - self._last_plain_status >= 60.0:
            self._last_plain_status = now
            total = f'/{self.total}' if self.total else ''
            _print(f'[eval] episodes={self.completed}{total} '
                   f'elapsed={_fmt_time(time.monotonic() - self.started)}')


def _record_episode_if_present(raw: str, dash: _Dashboard) -> None:
    match = _EPISODE_RE.match(raw)
    if not match:
        return
    dash.completed += 1
    try:
        dash.last_return = float(match.group(3))
        dash.returns.append(dash.last_return)
    except ValueError:
        pass


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description='Run checkpoint evaluation via Olympus.')
    ap.add_argument('--checkpoint', default=None,
                    help='checkpoint for legacy config mode')
    ap.add_argument('--config', default=None,
                    help='training config or kind: olympus-eval manifest')
    ap.add_argument('--episodes', type=int, default=None,
                    help='legacy mode episode override')
    ap.add_argument('--n-parallel', type=int, default=None)
    ap.add_argument('--environment', default=None,
                    help='legacy environment setup name or YAML path')
    ap.add_argument('--env-type', default=None,
                    help='legacy environment backend type')
    ap.add_argument('--python', default=None)
    ap.add_argument('--output-root', default=None)
    ap.add_argument('--run-name', default=None)
    ap.add_argument('-v', '--verbose', action='store_true')
    return ap.parse_args()


def _read_yaml(path):
    with open(path) as handle:
        value = yaml.safe_load(handle) or {}
    if not isinstance(value, dict):
        raise SystemExit(f'[eval] config is not a mapping: {path}')
    return value


def _reject_mixed_training_config(cfg):
    mixed = cfg.get('experience_collection') or {}
    if isinstance(mixed, dict) and mixed.get('enabled'):
        raise SystemExit(
            '[eval] experience_collection is a learner-side training feature; '
            'use an olympus-eval manifest with separate environments and matrix entries')


def _legacy_run(args):
    if not args.checkpoint:
        raise SystemExit('[eval] --checkpoint is required for a legacy config')
    checkpoint = os.path.abspath(args.checkpoint)
    if not os.path.exists(checkpoint):
        raise SystemExit(f'[eval] checkpoint not found: {checkpoint}')
    loaded_from_checkpoint = False
    if args.config:
        source = os.path.abspath(args.config)
        cfg = _read_yaml(source)
    else:
        cfg, source = load_checkpoint_config(checkpoint)
        if not cfg:
            raise SystemExit('[eval] no saved config found; pass --config')
        loaded_from_checkpoint = True
    _reject_mixed_training_config(cfg)
    cfg = copy.deepcopy(cfg)
    outputs = dict(cfg.get('outputs') or {})
    for key in ('run_dir', 'checkpoints_dir', 'episodes_dir', 'plots_dir',
                'telemetry_dir', 'traces_dir', 'resolved_config'):
        outputs.pop(key, None)
    if args.output_root:
        outputs['root'] = args.output_root
    if args.run_name:
        outputs['run_name'] = args.run_name
    elif loaded_from_checkpoint and not outputs.get('run_name'):
        alg = (cfg.get('runtime') or {}).get('algorithm', 'policy')
        outputs['run_name'] = f'eval_{alg}_{time.strftime("%Y%m%d-%H%M%S")}'
    cfg['outputs'] = outputs
    cfg.setdefault('training', {})['resume_from'] = checkpoint
    cfg['eval'] = {**(cfg.get('eval') or {}), 'enabled': True,
                   'checkpoint': checkpoint, 'logging': 'standard'}
    if args.episodes is not None:
        cfg['episodes'] = int(args.episodes)
        cfg.setdefault('orchestrator', {})['episodes'] = int(args.episodes)
    if args.n_parallel is not None:
        cfg['n_parallel'] = max(1, int(args.n_parallel))
        cfg.setdefault('orchestrator', {})['n_parallel'] = cfg['n_parallel']
    return [{
        'name': args.run_name or 'legacy', 'cfg': cfg, 'checkpoint': checkpoint,
        'source': source, 'logging': 'standard', 'episodes': args.episodes,
        'environment_override': args.environment, 'env_type_override': args.env_type,
        'matrix_index': 0,
    }]


def _manifest_runs(args):
    try:
        specs, _ = load_manifest(args.config)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        raise SystemExit(f'[eval] invalid manifest: {exc}') from exc
    runs = []
    for spec in specs:
        cp = spec['checkpoint_spec']
        checkpoint = cp['path']
        if not os.path.exists(checkpoint):
            runs.append({**spec, 'preflight_error': f'checkpoint not found: {checkpoint}'})
            continue
        cfg, source = load_checkpoint_config(checkpoint)
        if not cfg and cp.get('model_config'):
            source = cp['model_config']
            cfg = _read_yaml(source)
        if not cfg:
            runs.append({**spec, 'preflight_error':
                         'checkpoint has no neighboring resolved config and no model_config'})
            continue
        cfg = copy.deepcopy(cfg)
        scenario_path = spec['scenario_spec']['path']
        env_spec = spec['environment_spec']
        try:
            scenario = _read_yaml(scenario_path)
            validate_scenario(env_spec['type'], scenario)
            points = scenario_episode_count(scenario_path)
        except (OSError, ValueError) as exc:
            runs.append({**spec, 'preflight_error': str(exc)})
            continue

        # Remove training collection semantics and stale output paths. Model-
        # facing blocks stay intact for the existing checkpoint compatibility path.
        cfg.pop('experience_collection', None)
        cfg['environment'] = {'type': env_spec['type'], 'path': scenario_path}
        for key, value in (env_spec.get('options') or {}).items():
            cfg[key] = copy.deepcopy(value)
            if key in (cfg.get('orchestrator') or {}):
                cfg['orchestrator'][key] = copy.deepcopy(value)
        cfg['n_parallel'] = spec['n_parallel']
        cfg['episodes'] = points * spec['repetitions']
        cfg.setdefault('orchestrator', {})['n_parallel'] = cfg['n_parallel']
        cfg['orchestrator']['episodes'] = cfg['episodes']
        cfg['seed'] = spec['seed']
        cfg.setdefault('training', {})['resume_from'] = checkpoint
        cfg['eval'] = {'enabled': True, 'checkpoint': checkpoint,
                       'logging': spec['logging']}
        outputs = dict(cfg.get('outputs') or {})
        for key in ('run_dir', 'checkpoints_dir', 'episodes_dir', 'plots_dir',
                    'telemetry_dir', 'traces_dir', 'resolved_config'):
            outputs.pop(key, None)
        output_root = args.output_root or spec.get('output_root', 'olympus/evaluations')
        outputs.update({'root': output_root, 'run_name': spec['name']})
        if spec['logging'] != 'standard':
            outputs.update({'plot_episodes': False, 'require_state_logs': False})
        cfg['outputs'] = outputs
        cfg['eval_metadata'] = {
            'checkpoint_name': spec['checkpoint'], 'checkpoint_label': cp['label'],
            'scenario': spec['scenario'], 'environment': spec['environment'],
            'backend': env_spec['type'], 'repetitions': spec['repetitions'],
            'seed': spec['seed'], 'metadata': cp.get('metadata') or {},
        }
        runs.append({**spec, 'cfg': cfg, 'checkpoint': checkpoint,
                     'source': source, 'episodes': cfg['episodes']})
    return runs


def _write_temp_config(cfg: dict) -> str:
    fd, path = tempfile.mkstemp(prefix='olympus_eval_', suffix='.yaml')
    with os.fdopen(fd, 'w') as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False)
    return path


def _run_one(run, args, matrix_position, matrix_total):
    if run.get('preflight_error'):
        _print(f'{C.RED}✗{C.RESET} [{matrix_position}/{matrix_total}] '
               f'{run.get("name", "run")} preflight: {run["preflight_error"]}')
        return 2, 0, 0.0
    cfg, checkpoint = run['cfg'], run['checkpoint']
    env_type = str((cfg.get('environment') or {}).get('type') or 'mininet')
    if hasattr(os, 'geteuid') and os.geteuid() != 0 and env_type != 'raynet':
        _print(f'{C.RED}✗{C.RESET} [{matrix_position}/{matrix_total}] {run["name"]}: '
               'Mininet evaluation requires sudo -E')
        return 2, 0, 0.0

    profile = run.get('logging', 'standard')
    cleanup_root = None
    if profile == 'none':
        cleanup_root = tempfile.mkdtemp(prefix='olympus_eval_none_')
        cfg['outputs']['root'] = cleanup_root
    temp_config = _write_temp_config(cfg)
    python_bin = os.path.abspath(args.python or (cfg.get('paths') or {}).get('py', sys.executable))
    command = [python_bin, str(_HERE / 'orchestrator.py'), '--config', temp_config,
               '--eval', '--checkpoint', checkpoint]
    if run.get('environment_override'):
        command += ['--environment', run['environment_override']]
    if run.get('env_type_override'):
        command += ['--env-type', run['env_type_override']]

    log_path = None
    log_handle = None
    if profile == 'standard':
        _LOG_DIR.mkdir(parents=True, exist_ok=True)
        log_path = _LOG_DIR / f'eval_{time.strftime("%Y%m%d-%H%M%S")}_{matrix_position}.log'
        log_handle = log_path.open('w')
    rule = C.GREY + '─' * min(_term_width(), 78) + C.RESET
    runtime = cfg.get('runtime') or {}
    _print(rule)
    _print(f'{C.BOLD}{C.BLUE}eval{C.RESET} [{matrix_position}/{matrix_total}] '
           f'{C.BOLD}{run["name"]}{C.RESET} · {runtime.get("algorithm", "?")} · {env_type}')
    _print(f'  checkpoint  {C.GREY}{checkpoint}{C.RESET}')
    _print(f'  episodes    {run.get("episodes") or "unbounded"} · slots={cfg.get("n_parallel", 1)} '
           f'· files={profile}')
    _print(rule)

    child_env = dict(os.environ, PYTHONUNBUFFERED='1')
    proc = subprocess.Popen(command, cwd=str(_ROOT), env=child_env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1, errors='replace')
    dash = _EvalDashboard(run.get('episodes'))
    out_q = queue.Queue()
    threading.Thread(target=_reader, args=(proc, out_q), daemon=True).start()
    interrupts = 0
    try:
        while True:
            try:
                line = out_q.get(timeout=0.5)
            except queue.Empty:
                dash.render()
                continue
            if line is None:
                break
            if log_handle:
                log_handle.write(line)
            stripped = line.rstrip('\n')
            if args.verbose:
                _print(stripped)
                _record_episode_if_present(stripped, dash)
            else:
                _handle_line(stripped, dash)
    except KeyboardInterrupt:
        interrupts = 1
        proc.send_signal(signal.SIGINT)
    finally:
        returncode = proc.wait()
        dash.close()
        if log_handle:
            log_handle.close()
        try:
            os.unlink(temp_config)
        except OSError:
            pass
        if cleanup_root:
            shutil.rmtree(cleanup_root, ignore_errors=True)
    elapsed = time.monotonic() - dash.started
    ok = returncode == 0 and not interrupts
    _print(f'{C.GREEN if ok else C.RED}{"✓" if ok else "✗"}{C.RESET} '
           f'{run["name"]}: {dash.completed} episode(s) in {_fmt_time(elapsed)}'
           + (f' · log={log_path}' if log_path else ''))
    return (returncode or (130 if interrupts else 0)), dash.completed, elapsed


def main() -> int:
    args = _parse_args()
    if not args.config and not args.checkpoint:
        raise SystemExit('[eval] pass --config and/or --checkpoint')
    manifest = False
    if args.config:
        candidate = _read_yaml(os.path.abspath(args.config))
        manifest = is_eval_manifest(candidate)
    if manifest:
        if args.checkpoint or args.environment or args.env_type or args.episodes:
            raise SystemExit('[eval] checkpoint/environment/episodes CLI overrides are legacy-mode only')
        runs = _manifest_runs(args)
    else:
        runs = _legacy_run(args)

    started = time.monotonic()
    results = []
    for index, run in enumerate(runs, 1):
        code, completed, elapsed = _run_one(run, args, index, len(runs))
        results.append({'name': run.get('name', f'run-{index}'), 'code': code,
                        'completed': completed, 'elapsed_s': elapsed})
    failed = [result for result in results if result['code'] != 0]
    _print(C.GREY + '─' * min(_term_width(), 78) + C.RESET)
    _print(f'{C.BOLD}evaluation summary{C.RESET}  runs={len(results)} '
           f'passed={len(results)-len(failed)} failed={len(failed)} '
           f'elapsed={_fmt_time(time.monotonic()-started)}')
    for result in failed:
        _print(f'  {C.RED}✗{C.RESET} {result["name"]} exit={result["code"]}')
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())

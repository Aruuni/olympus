#!/usr/bin/env python3
"""Run deterministic checkpoint evaluation through the Olympus orchestrator.

This is intentionally a thin sibling of train.py: it resolves a config,
starts olympus/orchestrator.py, and leaves environment/slot/worker/state/reward
wiring to the existing architecture. The orchestrator runs in eval mode, so it
loads the requested checkpoint in each worker, disables exploration, and skips
learner startup/training.

Usage:
    ./venv_training/bin/python olympus/eval.py --checkpoint path/to/model.pt
    ./venv_training/bin/python olympus/eval.py --checkpoint path/to/model.pt --config olympus/config.yaml
"""

import argparse
import os
from pathlib import Path
import queue
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
from olympus.train import (
    C,
    _Dashboard,
    _EPISODE_RE,
    _fmt_time,
    _handle_line,
    _print,
    _reader,
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
    ap = argparse.ArgumentParser(
        description='Run checkpoint inference/evaluation via Olympus.')
    ap.add_argument('--checkpoint', required=True,
                    help='model checkpoint to evaluate')
    ap.add_argument('--config', default=None,
                    help='config yaml to use; defaults to the checkpoint run config when found')
    ap.add_argument('--episodes', type=int, default=None,
                    help='override episodes from the config')
    ap.add_argument('--n-parallel', type=int, default=None,
                    help='override n_parallel from the config')
    ap.add_argument('--environment', default=None,
                    help='environment_setup name or yaml path passed through')
    ap.add_argument('--env-type', default=None,
                    help='environment backend type passed through, e.g. raynet or mininet')
    ap.add_argument('--python', default=None,
                    help='interpreter for the orchestrator; defaults to paths.py from the config')
    ap.add_argument('--output-root', default=None,
                    help='override outputs.root for this eval run')
    ap.add_argument('--run-name', default=None,
                    help='override outputs.run_name for this eval run')
    ap.add_argument('-v', '--verbose', action='store_true',
                    help='stream raw orchestrator output instead of the condensed interface')
    return ap.parse_args()


def _load_eval_config(args: argparse.Namespace):
    checkpoint = os.path.abspath(args.checkpoint)
    if not os.path.exists(checkpoint):
        raise SystemExit(f'[eval] checkpoint not found: {checkpoint}')

    source = ''
    loaded_from_checkpoint = False
    if args.config:
        source = os.path.abspath(args.config)
        with open(source) as f:
            cfg = yaml.safe_load(f) or {}
    else:
        cfg, source = load_checkpoint_config(checkpoint)
        if not cfg:
            raise SystemExit(
                '[eval] could not find telemetry/config.resolved.yaml for '
                f'{checkpoint}; pass --config explicitly')
        loaded_from_checkpoint = True

    if not isinstance(cfg, dict):
        raise SystemExit(f'[eval] config is not a mapping: {source}')

    cfg = dict(cfg)
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
    cfg['eval'] = {
        **(cfg.get('eval') or {}),
        'enabled': True,
        'checkpoint': checkpoint,
    }
    if args.episodes is not None:
        cfg['episodes'] = int(args.episodes)
    if args.n_parallel is not None:
        cfg['n_parallel'] = max(1, int(args.n_parallel))

    return cfg, source, checkpoint


def _write_temp_config(cfg: dict) -> str:
    fd, path = tempfile.mkstemp(prefix='olympus_eval_', suffix='.yaml')
    with os.fdopen(fd, 'w') as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return path


def main() -> int:
    args = _parse_args()
    cfg, source_config, checkpoint = _load_eval_config(args)

    env_type = (args.env_type or
                ((cfg.get('environment', {}) or {}).get('type')) or
                'mininet')
    if (hasattr(os, 'geteuid') and os.geteuid() != 0 and
            str(env_type).lower() != 'raynet'):
        raise SystemExit(
            '[eval] evaluation requires root for Mininet; invoke this script '
            'with sudo -E and the project virtualenv Python')

    python_bin = os.path.abspath(
        args.python or (cfg.get('paths', {}) or {}).get('py', sys.executable))
    total_episodes = args.episodes or cfg.get('episodes')
    runtime = cfg.get('runtime', {}) or {}
    temp_config = _write_temp_config(cfg)

    command = [
        python_bin,
        str(_HERE / 'orchestrator.py'),
        '--config', temp_config,
        '--eval',
        '--checkpoint', checkpoint,
    ]
    if args.episodes is not None:
        command += ['--episodes', str(args.episodes)]
    if args.environment:
        command += ['--environment', args.environment]
    if args.env_type:
        command += ['--env-type', args.env_type]

    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime('%Y%m%d-%H%M%S')
    log_path = _LOG_DIR / f'eval_{stamp}.log'

    rule = C.GREY + '─' * min(_term_width(), 70) + C.RESET
    _print(rule)
    _print(f'{C.BOLD}{C.BLUE}eval{C.RESET}  '
           f'{C.BOLD}{runtime.get("algorithm", "?")}{C.RESET}'
           f'{C.DIM} · {runtime.get("reward", "?")}'
           f' · {runtime.get("state", "?")}'
           f' · {(cfg.get("environment", {}) or {}).get("name", "config")}'
           f'{C.RESET}')
    _print(f'  {C.CYAN}episodes{C.RESET}    '
           f'{total_episodes if total_episodes else "unbounded"}')
    _print(f'  {C.CYAN}checkpoint{C.RESET}  {C.GREY}{checkpoint}{C.RESET}')
    _print(f'  {C.CYAN}config{C.RESET}      {C.GREY}{source_config}{C.RESET}')
    _print(f'  {C.CYAN}log{C.RESET}         {C.GREY}{log_path}{C.RESET}')
    _print(rule)

    child_env = dict(os.environ)
    child_env['PYTHONUNBUFFERED'] = '1'
    proc = subprocess.Popen(
        command, cwd=str(_ROOT), env=child_env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, errors='replace')

    dash = _EvalDashboard(total_episodes)
    out_q = queue.Queue()
    threading.Thread(target=_reader, args=(proc, out_q), daemon=True).start()

    interrupts = 0
    with log_path.open('w') as log:
        while True:
            try:
                try:
                    line = out_q.get(timeout=0.5)
                except queue.Empty:
                    dash.render()
                    continue
                if line is None:
                    break
                log.write(line)
                stripped = line.rstrip('\n')
                if args.verbose:
                    _print(stripped)
                    _record_episode_if_present(stripped, dash)
                else:
                    _handle_line(stripped, dash)
            except KeyboardInterrupt:
                interrupts += 1
                if interrupts == 1:
                    dash.event('interrupt - waiting for graceful shutdown '
                               '(Ctrl-C again to kill)',
                               symbol='⚠', color=C.YELLOW)
                    try:
                        proc.send_signal(signal.SIGINT)
                    except Exception:
                        pass
                else:
                    proc.kill()

    returncode = proc.wait()
    dash.close()

    elapsed = time.monotonic() - dash.started
    _print(rule)
    if returncode == 0 and interrupts == 0:
        symbol, label = f'{C.GREEN}✓{C.RESET}', f'{C.GREEN}done{C.RESET}'
    elif interrupts:
        symbol, label = f'{C.YELLOW}⚠{C.RESET}', f'{C.YELLOW}interrupted{C.RESET}'
    else:
        symbol, label = (f'{C.RED}✗{C.RESET}',
                         f'{C.RED}{C.BOLD}exit {returncode}{C.RESET}')
    _print(f'{symbol} {C.BOLD}evaluation{C.RESET} {label} '
           f'{C.GREY}after {_fmt_time(elapsed)} · '
           f'{dash.completed} episode(s) · log: {log_path.name}{C.RESET}')
    return returncode


if __name__ == '__main__':
    raise SystemExit(main())

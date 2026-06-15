#!/usr/bin/env python3
"""Run a training session with a condensed live progress interface.

Wraps olympus/orchestrator.py the same way benchmarks/run_all.py wraps the
benchmark runners: the verbose child output is condensed into an in-place
status block (episode progress, ETA, returns, learner telemetry, checkpoint
age) plus permanent timestamped event lines for resumes, learner restarts,
and warnings. The full raw output is preserved in olympus/logs/ so nothing is
lost. Use --verbose to stream the raw child output instead.

Usage:
    sudo -E ./venv_training/bin/python olympus/train.py --config olympus/config.yaml
"""

import argparse
import collections
import os
from pathlib import Path
import queue
import re
import shutil
import signal
import subprocess
import sys
import threading
import time

import yaml

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
_LOG_DIR = _HERE / 'logs'

# ── Line patterns emitted by the orchestrator / learners ────────────────────

# Forwarded learner lines arrive as "[learner] [learner] step=..."; strip every
# leading "[tag] " so one pattern serves both direct and forwarded lines.
_TAG_PREFIX_RE = re.compile(r'^(?:\[[^\]]*\]\s+)+')
_EPISODE_RE = re.compile(
    r'^\[orch\] ep=(\d+)\s+env=(\S*)\s+type=\S+\s+return=(\S+)\s+elapsed=')
_LEARNER_STEP_RE = re.compile(r'^step=\s*(\d+)\s+(.*)$')
_CKPT_SAVE_RE = re.compile(r'\bsaved\b.*\bstep=(\d+)')
_EPISODE_CKPT_RE = re.compile(
    r'^\[orch\] saved episode checkpoint completed=(\d+)\s+path=(\S+)')
_RESUME_RE = re.compile(r'\bloaded\b.*\bstep=(\d+)')
_RUN_DIR_RE = re.compile(r'^\[orch\] run_dir=(\S+)')
_N_PARALLEL_RE = re.compile(r'^\[orch\] n_parallel=(\d+)')
_MODE_RE = re.compile(r'^\[orch\] mode=(\S+)\s+environment=(\S+)\s+pool=(\d+)')
_WARN_RE = re.compile(
    r'learner died|restart failed|non-finite|Traceback \(most recent call',
    re.IGNORECASE)
_INFO_EVENT_RE = re.compile(
    r'^\[orch\] (learner ready|learner restarted|using model config|'
    r'MARL collection ceiling|episode checkpoint copies)')


# ── Terminal / color helpers (matches benchmarks/run_all.py) ────────────────

class C:
    """ANSI styles, auto-disabled when not writing to a TTY or NO_COLOR set."""
    enabled = sys.stdout.isatty() and os.environ.get('NO_COLOR') is None
    RESET = '\033[0m' if enabled else ''
    BOLD = '\033[1m' if enabled else ''
    DIM = '\033[2m' if enabled else ''
    RED = '\033[31m' if enabled else ''
    GREEN = '\033[32m' if enabled else ''
    YELLOW = '\033[33m' if enabled else ''
    BLUE = '\033[34m' if enabled else ''
    CYAN = '\033[36m' if enabled else ''
    GREY = '\033[90m' if enabled else ''


def _fmt_time(seconds: float) -> str:
    seconds = int(seconds)
    if seconds < 60:
        return f'{seconds}s'
    if seconds < 3600:
        return f'{seconds // 60}m{seconds % 60:02d}s'
    return f'{seconds // 3600}h{(seconds % 3600) // 60:02d}m'


def _term_width() -> int:
    try:
        return shutil.get_terminal_size((100, 24)).columns
    except OSError:
        return 100


def _clip(text: str, width: int) -> str:
    return text if len(text) <= width else text[:max(0, width - 1)] + '…'


def _print(msg: str = '') -> None:
    print(msg, flush=True)


# ── Live dashboard ───────────────────────────────────────────────────────────

class _Dashboard:
    """In-place multi-line status block with permanent event lines above it.

    Tracks completed episodes against the configured total, the most recent
    returns, the learner's last telemetry line, and checkpoint times (shown
    inline in the stats line). Events (resumes, learner restarts, warnings)
    scroll above the block so their wall-clock times stay visible.
    """

    def __init__(self, total_episodes):
        self.total = total_episodes
        self.completed = 0
        self.returns = collections.deque(maxlen=50)
        self.last_return = None
        self.learner_step = None
        self.learner_tail = ''
        self.last_ckpt_step = None
        self.last_ckpt_time = None
        self.episode_ckpts = 0
        self.n_parallel = None
        self.started = time.monotonic()
        self.active = C.enabled
        self._block_lines = 0
        self._last_render = 0.0
        self._last_plain_status = 0.0

    # -- permanent output ----------------------------------------------------

    def event(self, text: str, symbol: str = '·', color: str = '') -> None:
        stamp = time.strftime('%H:%M:%S')
        line = (f'{C.GREY}{stamp}{C.RESET}  {color}{symbol}{C.RESET} '
                f'{color}{text}{C.RESET}')
        if self.active:
            self._erase_block()
            _print(line)
            self.render(force=True)
        else:
            _print(f'{stamp}  {symbol} {text}')

    def checkpoint(self, step: int) -> None:
        # Rolling saves arrive every save_every steps; the live stats line
        # ('ckpt step=N (age ago)') is enough — no permanent scroll line.
        if step == self.last_ckpt_step:
            return
        self.last_ckpt_step = step
        self.last_ckpt_time = time.monotonic()
        self.render(force=True)

    # -- live block ----------------------------------------------------------

    def _erase_block(self) -> None:
        if self._block_lines:
            sys.stdout.write(f'\033[{self._block_lines}A\033[J')
            self._block_lines = 0

    def _bar_line(self, width: int) -> str:
        elapsed = time.monotonic() - self.started
        if self.total:
            frac = max(0.0, min(1.0, self.completed / self.total))
            counter = f'{self.completed}/{self.total}'
            pct = f'{int(frac * 100):3d}%'
            rate = self.completed / elapsed if elapsed > 0 else 0.0
            eta = (f'ETA {_fmt_time((self.total - self.completed) / rate)}'
                   if rate > 0 and self.completed else 'ETA ...')
        else:
            frac, counter, pct, eta = 0.0, str(self.completed), '', ''
        head = f'{C.BOLD}{C.CYAN}episodes{C.RESET}'
        tail = (f'  {C.GREEN}{pct}{C.RESET}  {C.GREY}{eta}'
                f'  elapsed {_fmt_time(elapsed)}{C.RESET}')
        plain = len(f'episodes  {counter:>11}  {pct}  {eta}'
                    f'  elapsed {_fmt_time(elapsed)}') + 4
        bar_w = max(10, min(40, width - plain))
        filled = int(bar_w * frac)
        bar = (C.GREEN + '█' * filled + C.GREY + '░' * (bar_w - filled)
               + C.RESET)
        return f'{head}  {counter:>11}  {bar}{tail}'

    def _stats_line(self, width: int) -> str:
        parts = []
        if self.last_return is not None:
            parts.append(f'return last={self.last_return:.1f}')
        if self.returns:
            parts.append(
                f'avg{len(self.returns)}='
                f'{sum(self.returns) / len(self.returns):.1f}')
        elapsed = time.monotonic() - self.started
        if self.completed and elapsed > 0:
            parts.append(f'ep/h {self.completed / elapsed * 3600.0:.1f}')
        if self.n_parallel:
            parts.append(f'slots {self.n_parallel}')
        if self.last_ckpt_step is not None:
            age = _fmt_time(time.monotonic() - self.last_ckpt_time)
            ckpt = f'ckpt step={self.last_ckpt_step} ({age} ago)'
            if self.episode_ckpts:
                ckpt += f' +{self.episode_ckpts} episode copies'
            parts.append(ckpt)
        else:
            parts.append('ckpt none yet')
        text = '  ·  '.join(parts)
        return f'{C.GREY}{_clip(text, width - 2)}{C.RESET}'

    def _learner_line(self, width: int) -> str:
        if self.learner_step is None:
            return f'{C.GREY}learner   waiting for first update...{C.RESET}'
        head = f'{C.BOLD}{C.BLUE}learner{C.RESET}  step={self.learner_step}'
        tail = _clip(self.learner_tail, width - len(
            f'learner  step={self.learner_step}') - 4)
        return f'{head}  {C.DIM}{tail}{C.RESET}'

    def render(self, force: bool = False) -> None:
        if not self.active:
            # Non-TTY: emit a plain status line once a minute instead.
            now = time.monotonic()
            if force or now - self._last_plain_status >= 60.0:
                self._last_plain_status = now
                total = f'/{self.total}' if self.total else ''
                _print(f'[train] episodes={self.completed}{total} '
                       f'learner_step={self.learner_step} '
                       f'last_ckpt_step={self.last_ckpt_step} '
                       f'elapsed={_fmt_time(time.monotonic() - self.started)}')
            return
        now = time.monotonic()
        if not force and now - self._last_render < 0.25:
            return
        self._last_render = now
        width = _term_width()
        self._erase_block()
        lines = (self._bar_line(width), self._stats_line(width),
                 self._learner_line(width))
        for line in lines:
            sys.stdout.write('\r\033[K' + line + '\n')
        self._block_lines = len(lines)
        sys.stdout.flush()

    def close(self) -> None:
        if self.active:
            self._erase_block()
            sys.stdout.flush()


# ── Child line handling ──────────────────────────────────────────────────────

def _handle_line(raw: str, dash: _Dashboard) -> None:
    stripped = _TAG_PREFIX_RE.sub('', raw)

    match = _EPISODE_RE.match(raw)
    if match:
        dash.completed += 1
        try:
            dash.last_return = float(match.group(3))
            dash.returns.append(dash.last_return)
        except ValueError:
            pass
        dash.render()
        return

    if '[learner]' in raw:
        match = _LEARNER_STEP_RE.match(stripped)
        if match:
            dash.learner_step = int(match.group(1))
            dash.learner_tail = match.group(2).strip()
            dash.render()
            return
        if 'saved' in stripped and 'replay' not in stripped:
            match = _CKPT_SAVE_RE.search(stripped)
            if match:
                dash.checkpoint(int(match.group(1)))
                return
        if 'loaded' in stripped:
            match = _RESUME_RE.search(stripped)
            if match:
                dash.event(f'resumed from checkpoint step={match.group(1)}',
                           symbol='↻', color=C.CYAN)
                return

    match = _EPISODE_CKPT_RE.match(raw)
    if match:
        # Counted in the stats line ('+N episode copies'); no scroll line.
        dash.episode_ckpts += 1
        dash.render()
        return

    match = _RUN_DIR_RE.match(raw)
    if match:
        dash.event(f'run dir  {match.group(1)}', symbol='·', color=C.CYAN)
        return
    match = _N_PARALLEL_RE.match(raw)
    if match:
        dash.n_parallel = int(match.group(1))
        return
    match = _MODE_RE.match(raw)
    if match:
        dash.event(
            f'mode {match.group(1)}  environment {match.group(2)}  '
            f'pool {match.group(3)}', symbol='·', color=C.CYAN)
        return
    if _WARN_RE.search(raw):
        dash.event(_clip(raw, _term_width() - 14), symbol='⚠', color=C.YELLOW)
        return
    if _INFO_EVENT_RE.match(raw):
        dash.event(_clip(raw[len('[orch] '):], _term_width() - 14),
                   symbol='·', color='')


# ── Entry point ──────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description='Train with a condensed live progress interface.')
    ap.add_argument('--config', default=str(_HERE / 'config.yaml'))
    ap.add_argument('--episodes', type=int, default=None,
                    help='override episodes from the config')
    ap.add_argument('--environment', default=None,
                    help='environment name or yaml path passed through')
    ap.add_argument('--python', default=None,
                    help='interpreter for the orchestrator; '
                         'defaults to paths.py from the config')
    ap.add_argument('-v', '--verbose', action='store_true',
                    help='stream raw orchestrator output instead of the '
                         'condensed interface (still logged)')
    return ap.parse_args()


def _reader(proc, out_q):
    for line in proc.stdout:
        out_q.put(line)
    out_q.put(None)


def main() -> int:
    args = _parse_args()
    config_path = os.path.abspath(args.config)
    with open(config_path) as f:
        cfg = yaml.safe_load(f) or {}

    if hasattr(os, 'geteuid') and os.geteuid() != 0:
        raise SystemExit(
            '[train] training requires root for Mininet; invoke this script '
            'with sudo -E and the project virtualenv Python')

    python_bin = os.path.abspath(
        args.python or (cfg.get('paths', {}) or {}).get('py', sys.executable))
    total_episodes = args.episodes or cfg.get('episodes')
    runtime = cfg.get('runtime', {}) or {}

    command = [python_bin, str(_HERE / 'orchestrator.py'),
               '--config', config_path]
    if args.episodes is not None:
        command += ['--episodes', str(args.episodes)]
    if args.environment:
        command += ['--environment', args.environment]

    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime('%Y%m%d-%H%M%S')
    log_path = _LOG_DIR / f'train_{stamp}.log'

    rule = C.GREY + '─' * min(_term_width(), 70) + C.RESET
    _print(rule)
    _print(f'{C.BOLD}{C.BLUE}train{C.RESET}  '
           f'{C.BOLD}{runtime.get("algorithm", "?")}{C.RESET}'
           f'{C.DIM} · {runtime.get("reward", "?")}'
           f' · {runtime.get("state", "?")}'
           f' · {(cfg.get("environment", {}) or {}).get("name", "config")}'
           f'{C.RESET}')
    _print(f'  {C.CYAN}episodes{C.RESET}  '
           f'{total_episodes if total_episodes else "unbounded"}')
    _print(f'  {C.CYAN}log{C.RESET}       {C.GREY}{log_path}{C.RESET}')
    _print(rule)

    child_env = dict(os.environ)
    child_env['PYTHONUNBUFFERED'] = '1'
    proc = subprocess.Popen(
        command, cwd=str(_ROOT), env=child_env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, errors='replace')

    dash = _Dashboard(total_episodes)
    out_q = queue.Queue()
    threading.Thread(target=_reader, args=(proc, out_q), daemon=True).start()

    interrupts = 0
    with log_path.open('w') as log:
        while True:
            try:
                try:
                    line = out_q.get(timeout=0.5)
                except queue.Empty:
                    dash.render()  # keep elapsed / ETA ticking
                    continue
                if line is None:
                    break
                log.write(line)
                stripped = line.rstrip('\n')
                if args.verbose:
                    _print(stripped)
                else:
                    _handle_line(stripped, dash)
            except KeyboardInterrupt:
                # The orchestrator shares the foreground process group, so it
                # received SIGINT too and starts a graceful shutdown (final
                # checkpoint save). Keep draining its output; a second Ctrl-C
                # kills it hard.
                interrupts += 1
                if interrupts == 1:
                    dash.event('interrupt - waiting for graceful shutdown '
                               '(Ctrl-C again to kill)',
                               symbol='⚠', color=C.YELLOW)
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
    ckpt = (f'last checkpoint step={dash.last_ckpt_step}'
            if dash.last_ckpt_step is not None else 'no checkpoint saved')
    _print(f'{symbol} {C.BOLD}training{C.RESET} {label} '
           f'{C.GREY}after {_fmt_time(elapsed)} · '
           f'{dash.completed} episode(s) · {ckpt} · log: {log_path.name}'
           f'{C.RESET}')
    return returncode


if __name__ == '__main__':
    raise SystemExit(main())

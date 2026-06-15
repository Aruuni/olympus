#!/usr/bin/env python3
"""Run configured approaches through every top-level benchmark suite.

Output is condensed to a colored progress bar per benchmark; the full, verbose
output of each child runner is preserved in benchmarks/logs/ so nothing is lost.
Use --verbose to stream the raw child output instead of the progress bar.
"""

import argparse
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import time

import yaml


_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
_APPROACHES_CONFIG = _HERE / 'config.yaml'
_LOG_DIR = _HERE / 'logs'

_BENCHMARKS = (
    (
        'responsiveness',
        _HERE / 'benchmark_responsiveness' / 'responsiveness.py',
        _HERE / 'benchmark_responsiveness' / 'config.yaml',
    ),
    (
        'responsive-fairness',
        _HERE / 'benchmark_responsive_fairness' / 'responsive_fairness.py',
        _HERE / 'benchmark_responsive_fairness' / 'config.yaml',
    ),
    (
        'fairness',
        _HERE / 'benchmark_fairness' / 'fairness.py',
        _HERE / 'benchmark_fairness' / 'config.yaml',
    ),
    (
        'inter-rtt-fairness',
        _HERE / 'benchmark_inter_rtt_fairness' / 'inter_rtt_fairness.py',
        _HERE / 'benchmark_inter_rtt_fairness' / 'config.yaml',
    ),
    (
        'convergence-4flow',
        _HERE / 'benchmark_convergence_4flow' / 'convergence_4flow.py',
        _HERE / 'benchmark_convergence_4flow' / 'config.yaml',
    ),
)

# Progress markers emitted by every child runner, e.g. "... (12/40)".
_PROGRESS_RE = re.compile(r'\((\d+)/(\d+)\)')
# A line that signals a hard failure inside a child runner.
_ERROR_RE = re.compile(r'traceback|error|exception|failed', re.IGNORECASE)


# ── Terminal / color helpers ────────────────────────────────────────────────

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


def _print(msg: str = '') -> None:
    print(msg, flush=True)


class _Bar:
    """A single-line, in-place progress bar for one benchmark."""

    def __init__(self, name: str, index: int, count: int):
        self.name = name
        self.index = index
        self.count = count
        self.completed = 0
        self.total = 0
        self.run = ''
        self.phase = 0
        self.started = time.monotonic()
        self.active = C.enabled

    def update_from_line(self, line: str) -> bool:
        """Parse a child line; return True if it advanced the bar."""
        matches = _PROGRESS_RE.findall(line)
        if not matches:
            return False
        completed, total = (int(x) for x in matches[-1])
        # A child resets its (completed/total) counter for each new approach.
        if total != self.total or completed < self.completed:
            if self.total:
                self.phase += 1
            self.total = total
        self.completed = completed
        run = re.search(r'run=(\d+)/(\d+)', line)
        self.run = f'{run.group(1)}/{run.group(2)}' if run else ''
        return True

    def render(self) -> None:
        if not self.active:
            return
        head = f'{C.BOLD}{C.CYAN}[{self.index}/{self.count}] {self.name}{C.RESET}'
        frac = (self.completed / self.total) if self.total else 0.0
        frac = max(0.0, min(1.0, frac))
        pct = int(frac * 100)
        elapsed = _fmt_time(time.monotonic() - self.started)
        counter = f'{self.completed}/{self.total}' if self.total else '...'
        extras = []
        if self.run:
            extras.append(f'run {self.run}')
        if self.phase:
            extras.append(f'phase {self.phase + 1}')
        extra = ('  ' + C.GREY + ' '.join(extras) + C.RESET) if extras else ''
        # Reserve space for the textual parts, then size the bar to fit.
        prefix = (f'\r\033[K{head}  {counter:>9}  ')
        suffix = f'  {C.GREEN}{pct:3d}%{C.RESET}{extra}  {C.GREY}{elapsed}{C.RESET}'
        plain_len = (len(self.name) + len(f'[{self.index}/{self.count}] ')
                     + 11 + len(f'{pct:3d}%') + len(elapsed)
                     + sum(len(e) for e in extras) + 12)
        width = max(10, min(40, _term_width() - plain_len))
        filled = int(width * frac)
        bar = (C.GREEN + '█' * filled + C.GREY + '░' * (width - filled) + C.RESET)
        sys.stdout.write(prefix + bar + suffix)
        sys.stdout.flush()

    def finish(self, ok: bool, elapsed: float, detail: str = '') -> None:
        symbol = f'{C.GREEN}✓{C.RESET}' if ok else f'{C.RED}✗{C.RESET}'
        label = (f'{C.GREEN}done{C.RESET}' if ok
                 else f'{C.RED}{C.BOLD}FAILED{C.RESET}')
        tail = f'  {C.GREY}{detail}{C.RESET}' if detail else ''
        if self.active:
            sys.stdout.write('\r\033[K')
            sys.stdout.flush()
        _print(f'{symbol} {C.BOLD}[{self.index}/{self.count}] {self.name}{C.RESET} '
               f'{label} {C.GREY}in {_fmt_time(elapsed)}{C.RESET}{tail}')


# ── Approach / argument resolution (unchanged behavior) ──────────────────────

def _slug(value: object) -> str:
    text = re.sub(r'[^A-Za-z0-9_.-]+', '_', str(value or '').strip())
    return text.strip('_') or 'learner'


def _approach_label(approach: dict) -> str:
    for key in ('data_folder', 'data_dir', 'name'):
        if approach.get(key):
            return _slug(approach[key])
    return _slug(
        f"{approach.get('algorithm', 'algorithm')}_"
        f"{approach.get('state', 'state')}_"
        f"{approach.get('reward', 'reward')}_"
        f"{approach.get('environment', 'environment')}"
    )


def _configured_approaches() -> list:
    with _APPROACHES_CONFIG.open() as f:
        cfg = yaml.safe_load(f) or {}
    return [dict(raw or {}) for raw in (cfg.get('approaches') or [])]


def _resolve_approaches(requested: list = None) -> list:
    approaches = _configured_approaches()
    if requested:
        available = ', '.join(_approach_label(item) for item in approaches) or '(none)'
        resolved = []
        for value in requested:
            wanted = _slug(value)
            matches = []
            for approach in approaches:
                keys = {_approach_label(approach)}
                if approach.get('name'):
                    keys.add(_slug(approach['name']))
                if wanted in keys:
                    matches.append(approach)
            if len(matches) != 1:
                raise SystemExit(
                    f'[run_all] approach not found: {value}; '
                    f'available: {available}')
            label = _approach_label(matches[0])
            if label not in resolved:
                resolved.append(label)
        return resolved

    labels = [_approach_label(approach) for approach in approaches]
    if not labels:
        raise SystemExit(
            '[run_all] no approaches configured in benchmarks/config.yaml')
    return labels


def _parse_args() -> argparse.Namespace:
    names = [name for name, _, _ in _BENCHMARKS]
    ap = argparse.ArgumentParser(
        description='Run configured approaches sequentially through all benchmarks.')
    ap.add_argument(
        '--approach',
        action='append',
        help='only run this approach data_folder/name; repeatable; defaults to all')
    ap.add_argument(
        '--experiment', '--only',
        action='append',
        dest='experiment',
        choices=names,
        help='run only this benchmark; repeat to select more than one')
    ap.add_argument(
        '--dry-run',
        action='store_true',
        help='validate every selected benchmark without starting Mininet')
    ap.add_argument(
        '--keep-going',
        action='store_true',
        help='continue with later benchmarks after a failure')
    ap.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='stream raw child output instead of the condensed progress bar')
    return ap.parse_args()


# ── Benchmark execution ──────────────────────────────────────────────────────

def _run_benchmark(spec, approaches, args, bar: _Bar, log_path: Path) -> int:
    """Run one benchmark, driving the progress bar and writing a full log."""
    name, runner, config = spec
    command = [sys.executable, str(runner), '--config', str(config)]
    for approach in approaches:
        command.extend(['--approach', approach])
    if args.dry_run:
        command.append('--dry-run')

    if args.verbose:
        # Pass-through mode: stream everything, no bar.
        return subprocess.run(command, cwd=str(_ROOT)).returncode

    last_lines: list = []
    bar.render()
    with log_path.open('w') as log, subprocess.Popen(
        command, cwd=str(_ROOT), stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, bufsize=1,
    ) as proc:
        for line in proc.stdout:
            log.write(line)
            stripped = line.rstrip('\n')
            last_lines.append(stripped)
            if len(last_lines) > 30:
                last_lines.pop(0)
            if bar.update_from_line(stripped):
                bar.render()
        returncode = proc.wait()

    if returncode != 0:
        # Surface the tail of the child output so failures are diagnosable.
        sys.stdout.write('\r\033[K')
        for tail in last_lines[-8:]:
            color = C.RED if _ERROR_RE.search(tail) else C.GREY
            _print(f'  {color}{tail}{C.RESET}')
    return returncode


def main() -> int:
    args = _parse_args()
    approaches = _resolve_approaches(args.approach)
    selected = set(args.experiment or [])
    benchmarks = [
        spec for spec in _BENCHMARKS if not selected or spec[0] in selected
    ]

    if not args.dry_run and hasattr(os, 'geteuid') and os.geteuid() != 0:
        raise SystemExit(
            '[run_all] benchmark execution requires root; invoke this script with '
            'sudo -E and the project virtualenv Python')

    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime('%Y%m%d-%H%M%S')

    rule = C.GREY + '─' * min(_term_width(), 70) + C.RESET
    _print(rule)
    _print(f'{C.BOLD}{C.BLUE}run_all{C.RESET}  '
           f'{C.DIM}{len(benchmarks)} benchmark(s) × {len(approaches)} approach(es)'
           f'{" · dry-run" if args.dry_run else ""}{C.RESET}')
    _print(f'  {C.CYAN}approaches{C.RESET} {", ".join(approaches)}')
    _print(f'  {C.CYAN}benchmarks{C.RESET} {", ".join(n for n, _, _ in benchmarks)}')
    _print(f'  {C.CYAN}logs{C.RESET}       {C.GREY}{_LOG_DIR}{C.RESET}')
    _print(rule)

    failures = []
    total_start = time.monotonic()
    for index, spec in enumerate(benchmarks, start=1):
        name = spec[0]
        bar = _Bar(name, index, len(benchmarks))
        log_path = _LOG_DIR / f'{stamp}_{index:02d}_{name}.log'
        started = time.monotonic()
        returncode = _run_benchmark(spec, approaches, args, bar, log_path)
        elapsed = time.monotonic() - started

        if returncode == 0:
            bar.finish(True, elapsed)
        else:
            failures.append((name, returncode))
            bar.finish(False, elapsed,
                       f'exit {returncode} · log: {log_path.name}')
            if not args.keep_going:
                break

    total_elapsed = time.monotonic() - total_start
    _print(rule)
    if failures:
        details = ', '.join(f'{name} (exit {code})' for name, code in failures)
        _print(f'{C.RED}{C.BOLD}✗ failed{C.RESET} after '
               f'{C.BOLD}{_fmt_time(total_elapsed)}{C.RESET}: {C.RED}{details}{C.RESET}')
        return 1

    _print(f'{C.GREEN}{C.BOLD}✓ all {len(benchmarks)} benchmark(s) completed{C.RESET} '
           f'in {C.BOLD}{_fmt_time(total_elapsed)}{C.RESET}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

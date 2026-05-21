"""
Plot individual random-BW benchmark runs.

Given a random_bw_rtt_goodput_single_* output directory, this script rebuilds
the saved random BW/RTT schedules and overlays any per-protocol goodput traces
found under each run's csvs/x1.csv.

Outputs:
  random_bw_runs_overview.pdf        - all run BW schedules in one grid
  random_bw_runs_all.pdf             - multipage detailed run plots
  <protocol>/runN/run_plot.pdf       - one detailed plot per protocol run

The overview/all-runs PDFs stay at the benchmark root when possible. The
individual run plots are written beside each run's csvs/ and schedule.csv.

Example:
  ./venv_training/bin/python single_agent_olympus/plot_random_bw_runs.py \\
    single_agent_olympus/data/random_bw_rtt_goodput_single_20260501-022552
"""

import argparse
import csv
import glob
import json
import math
import os
import re
import sys
from collections import defaultdict

os.environ.setdefault('MPLCONFIGDIR', os.path.join('/tmp', f'matplotlib-{os.getuid()}'))
os.makedirs(os.environ['MPLCONFIGDIR'], exist_ok=True)

import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np


_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)

COLORS = {
    'capacity': 'black',
    'rtt': '#7f7f7f',
    'selected': '#4878cf',
    'bbr3': '#e15759',
    'cubic': '#59a14f',
    'bbr1': '#f28e2c',
    'orca': '#b07aa1',
}

PROTOCOL_ORDER = ['selected', 'bbr3', 'cubic', 'bbr1', 'orca']


def _finite_float(value, default=math.nan):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def _latest_run_dir():
    pattern = os.path.join(
        _ROOT, 'single_agent_olympus', 'data', 'random_bw_rtt_goodput_single_*')
    candidates = [p for p in glob.glob(pattern) if os.path.isdir(p)]
    if not candidates:
        raise SystemExit('[plot_random_bw] no random_bw_rtt_goodput_single_* directory found')
    return max(candidates, key=os.path.getmtime)


def _run_number_from_name(name):
    m = re.search(r'run(\d+)$', os.path.basename(name))
    return int(m.group(1)) if m else None


def _read_schedule_csv(path):
    schedule = []
    try:
        with open(path, newline='') as f:
            for row in csv.DictReader(f):
                t = _finite_float(row.get('t'))
                bw = _finite_float(row.get('bw_mbps', row.get('bw')))
                delay = _finite_float(row.get('rtt_ms', row.get('delay')))
                if math.isfinite(t) and math.isfinite(bw) and math.isfinite(delay):
                    schedule.append({'t': t, 'bw': bw, 'delay': delay})
    except FileNotFoundError:
        pass
    return sorted(schedule, key=lambda item: item['t'])


def _load_schedules(run_dir):
    schedules_path = os.path.join(run_dir, 'schedules.json')
    if os.path.exists(schedules_path):
        with open(schedules_path) as f:
            raw = json.load(f)
        schedules = {
            int(run): sorted(
                [{'t': _finite_float(x['t']),
                  'bw': _finite_float(x['bw']),
                  'delay': _finite_float(x['delay'])}
                 for x in sched],
                key=lambda item: item['t'],
            )
            for run, sched in raw.items()
        }
        return {run: sched for run, sched in schedules.items() if sched}

    schedules = {}
    for path in glob.glob(os.path.join(run_dir, '*', 'run*', 'schedule.csv')):
        run = _run_number_from_name(os.path.dirname(path))
        if run is None or run in schedules:
            continue
        sched = _read_schedule_csv(path)
        if sched:
            schedules[run] = sched
    return schedules


def _read_metrics(run_dir):
    metrics_path = os.path.join(run_dir, 'goodput_metrics.csv')
    rows = []
    try:
        with open(metrics_path, newline='') as f:
            rows = list(csv.DictReader(f))
    except FileNotFoundError:
        pass
    return rows


def _protocol_key(row):
    proto = (row.get('protocol') or '').strip()
    if proto == 'selected':
        return 'selected'
    return proto


def _protocol_label(row):
    proto = _protocol_key(row)
    if proto == 'selected':
        return row.get('label') or 'selected'
    if proto == 'bbr3':
        return 'BBR3'
    if proto == 'bbr1':
        return 'BBR1'
    if proto == 'cubic':
        return 'CUBIC'
    if proto == 'orca':
        return 'Orca'
    return proto or 'unknown'


def _protocol_sort_key(proto):
    try:
        return (PROTOCOL_ORDER.index(proto), proto)
    except ValueError:
        return (len(PROTOCOL_ORDER), proto)


def _discover_protocol_runs(run_dir, metrics_rows):
    by_run = defaultdict(dict)
    for row in metrics_rows:
        run = row.get('run')
        try:
            run = int(float(run))
        except (TypeError, ValueError):
            continue
        proto = _protocol_key(row)
        proto_run_dir = row.get('run_dir') or ''
        if not proto_run_dir:
            continue
        by_run[run][proto] = {
            'label': _protocol_label(row),
            'run_dir': proto_run_dir,
            'success': str(row.get('success', '')).strip() in ('1', '1.0', 'true', 'True'),
            'average_goodput_mbps': _finite_float(row.get('average_goodput_mbps')),
        }

    if by_run:
        return by_run

    for proto_dir in glob.glob(os.path.join(run_dir, '*')):
        if not os.path.isdir(proto_dir):
            continue
        proto_name = os.path.basename(proto_dir)
        if proto_name in ('selected_runtime', 'random_bw_run_plots'):
            continue
        proto = 'selected' if proto_name.startswith('selected_') else proto_name
        for candidate in glob.glob(os.path.join(proto_dir, 'run*')):
            run = _run_number_from_name(candidate)
            if run is None:
                continue
            by_run[run][proto] = {
                'label': proto_name.replace('selected_', '') if proto == 'selected' else proto,
                'run_dir': candidate,
                'success': True,
                'average_goodput_mbps': math.nan,
            }
    return by_run


def _read_trace(csv_path):
    t, bw = [], []
    try:
        with open(csv_path, newline='') as f:
            for row in csv.DictReader(f):
                tv = _finite_float(row.get('time'))
                bv = _finite_float(row.get('bandwidth'))
                if math.isfinite(tv) and math.isfinite(bv):
                    t.append(tv)
                    bw.append(bv)
    except FileNotFoundError:
        pass
    return np.asarray(t, dtype=float), np.asarray(bw, dtype=float)


def _step_xy(schedule, duration=None, key='bw'):
    if not schedule:
        return np.asarray([]), np.asarray([])
    times = [float(item['t']) for item in schedule]
    vals = [float(item[key]) for item in schedule]
    if duration is None:
        if len(times) > 1:
            duration = times[-1] + (times[1] - times[0])
        else:
            duration = times[-1] + 15.0
    times.append(float(duration))
    vals.append(vals[-1])
    return np.asarray(times, dtype=float), np.asarray(vals, dtype=float)


def _run_duration(schedule, protocol_items):
    duration = None
    for item in protocol_items.values():
        csv_path = os.path.join(item['run_dir'], 'csvs', 'x1.csv')
        t, _ = _read_trace(csv_path)
        if len(t):
            duration = max(duration or 0.0, float(np.nanmax(t)))
    if duration is not None:
        return duration
    if len(schedule) > 1:
        return schedule[-1]['t'] + (schedule[1]['t'] - schedule[0]['t'])
    return 300.0


def _format_avg(value):
    return f'{value:.1f} Mbps' if math.isfinite(value) else ''


def plot_one_run(run, schedule, protocol_items, output_path=None, all_pdf=None):
    duration = _run_duration(schedule, protocol_items)
    tx, bw = _step_xy(schedule, duration=duration, key='bw')
    _, rtt = _step_xy(schedule, duration=duration, key='delay')

    fig, axes = plt.subplots(
        2, 1, figsize=(9.0, 5.2), sharex=True,
        gridspec_kw={'height_ratios': [1.0, 1.35], 'hspace': 0.12},
    )
    fig.suptitle(f'Random BW/RTT run {run}', fontsize=12, fontweight='bold')

    ax = axes[0]
    ax.step(tx, bw, where='post', color=COLORS['capacity'], linewidth=1.4,
            label='capacity')
    ax.set_ylabel('BW (Mbps)')
    ax.set_ylim(0, max(105.0, float(np.nanmax(bw)) * 1.08 if len(bw) else 105.0))
    ax.grid(True, alpha=0.28, linewidth=0.5)

    ax_rtt = ax.twinx()
    ax_rtt.step(tx, rtt, where='post', color=COLORS['rtt'], linewidth=1.0,
                linestyle='--', label='RTT')
    ax_rtt.set_ylabel('RTT (ms)')
    ax_rtt.set_ylim(0, max(105.0, float(np.nanmax(rtt)) * 1.08 if len(rtt) else 105.0))

    lines, labels = ax.get_legend_handles_labels()
    rlines, rlabels = ax_rtt.get_legend_handles_labels()
    ax.legend(lines + rlines, labels + rlabels, loc='upper right',
              frameon=False, fontsize=8, ncol=2)

    ax = axes[1]
    ax.step(tx, bw, where='post', color='black', linewidth=0.9,
            alpha=0.4, label='capacity')
    plotted = 0
    for proto in sorted(protocol_items, key=_protocol_sort_key):
        item = protocol_items[proto]
        csv_path = os.path.join(item['run_dir'], 'csvs', 'x1.csv')
        t, goodput = _read_trace(csv_path)
        if not len(t):
            continue
        label = item['label']
        avg = _format_avg(item.get('average_goodput_mbps', math.nan))
        if avg:
            label = f'{label} ({avg})'
        ax.plot(t, goodput, color=COLORS.get(proto), linewidth=1.0,
                alpha=0.9, label=label)
        plotted += 1

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Goodput (Mbps)')
    ax.set_xlim(0, duration)
    ax.set_ylim(0, max(105.0, float(np.nanmax(bw)) * 1.12 if len(bw) else 105.0))
    ax.grid(True, alpha=0.28, linewidth=0.5)
    if plotted:
        ax.legend(loc='upper right', frameon=False, fontsize=7, ncol=2)
    else:
        ax.text(0.5, 0.5, 'No per-protocol csvs/x1.csv traces found',
                transform=ax.transAxes, ha='center', va='center', fontsize=10)

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, dpi=180, bbox_inches='tight')
    if all_pdf is not None:
        all_pdf.savefig(fig, dpi=180, bbox_inches='tight')
    plt.close(fig)


def plot_overview(schedules, output_path):
    runs = sorted(schedules)
    if not runs:
        return
    ncols = 5
    nrows = int(math.ceil(len(runs) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, max(2.0, nrows * 1.45)),
                             sharex=True, sharey=True)
    axes = np.asarray(axes).reshape(-1)
    for ax, run in zip(axes, runs):
        sched = schedules[run]
        duration = sched[-1]['t'] + (sched[1]['t'] - sched[0]['t']) if len(sched) > 1 else 300
        tx, bw = _step_xy(sched, duration=duration, key='bw')
        ax.step(tx, bw, where='post', color=COLORS['capacity'], linewidth=0.9)
        ax.set_title(f'run {run}', fontsize=8)
        ax.grid(True, alpha=0.22, linewidth=0.4)
        ax.tick_params(labelsize=6)
    for ax in axes[len(runs):]:
        ax.axis('off')
    fig.suptitle('Random BW schedules by run', fontsize=12, fontweight='bold')
    fig.text(0.5, 0.02, 'Time (s)', ha='center', va='center', fontsize=10)
    fig.text(0.02, 0.5, 'BW (Mbps)', ha='center', va='center',
             rotation='vertical', fontsize=10)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches='tight')
    plt.close(fig)


def _parse_run_subset(raw):
    if not raw:
        return None
    out = set()
    for part in str(raw).replace(',', ' ').split():
        if '-' in part:
            lo, hi = part.split('-', 1)
            out.update(range(int(lo), int(hi) + 1))
        else:
            out.add(int(part))
    return out


def _choose_plot_root(run_dir, requested=None):
    if requested:
        return os.path.abspath(requested)
    if os.access(run_dir, os.W_OK):
        return run_dir
    parent = os.path.dirname(run_dir)
    fallback = os.path.join(parent, os.path.basename(run_dir) + '_plots')
    return os.path.abspath(fallback)


def _write_run_dir_plots(schedules, by_run):
    generated = []
    skipped = []
    for run in sorted(schedules):
        for proto in sorted(by_run.get(run, {}), key=_protocol_sort_key):
            item = by_run[run][proto]
            output_path = os.path.join(item['run_dir'], 'run_plot.pdf')
            try:
                plot_one_run(
                    run,
                    schedules[run],
                    {proto: item},
                    output_path=output_path,
                )
                generated.append(output_path)
            except PermissionError as exc:
                skipped.append((output_path, str(exc)))
            except OSError as exc:
                skipped.append((output_path, str(exc)))
    return generated, skipped


def main():
    ap = argparse.ArgumentParser(description='Plot all individual random-BW benchmark runs.')
    ap.add_argument('run_dir', nargs='?', default=None,
                    help='random_bw_rtt_goodput_single_* directory; default is latest')
    ap.add_argument('--plot-root', default=None,
                    help='directory for overview/all-runs PDFs; default is run_dir, or sibling fallback if run_dir is not writable')
    ap.add_argument('--output-dir', default=None,
                    help='optional directory for overlay run PDFs; default writes no separate individual directory')
    ap.add_argument('--runs', default=None,
                    help='optional run subset, e.g. "1 2 5-10"')
    args = ap.parse_args()

    run_dir = os.path.abspath(args.run_dir or _latest_run_dir())
    if not os.path.isdir(run_dir):
        raise SystemExit(f'[plot_random_bw] run_dir not found: {run_dir}')

    schedules = _load_schedules(run_dir)
    metrics_rows = _read_metrics(run_dir)
    by_run = _discover_protocol_runs(run_dir, metrics_rows)
    wanted = _parse_run_subset(args.runs)
    if wanted is not None:
        schedules = {run: sched for run, sched in schedules.items() if run in wanted}

    if not schedules:
        raise SystemExit(f'[plot_random_bw] no schedules found in {run_dir}')

    plot_root = _choose_plot_root(run_dir, requested=args.plot_root)
    os.makedirs(plot_root, exist_ok=True)

    overview_pdf = os.path.join(plot_root, 'random_bw_runs_overview.pdf')
    all_pdf_path = os.path.join(plot_root, 'random_bw_runs_all.pdf')
    plot_overview(schedules, overview_pdf)

    run_dir_plots, skipped = _write_run_dir_plots(schedules, by_run)
    overlay_paths = []
    output_dir = os.path.abspath(args.output_dir) if args.output_dir else None
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with PdfPages(all_pdf_path) as all_pdf:
        for run in sorted(schedules):
            output_path = os.path.join(output_dir, f'run{run:03d}.pdf') if output_dir else None
            plot_one_run(
                run,
                schedules[run],
                by_run.get(run, {}),
                output_path,
                all_pdf=all_pdf,
            )
            if output_path:
                overlay_paths.append(output_path)

    print(f'[plot_random_bw] run_dir={run_dir}')
    print(f'[plot_random_bw] plot_root={plot_root}')
    print(f'[plot_random_bw] overview={overview_pdf}')
    print(f'[plot_random_bw] all_runs={all_pdf_path}')
    if output_dir:
        print(f'[plot_random_bw] overlay_dir={output_dir}')
        print(f'[plot_random_bw] generated_overlays={len(overlay_paths)}')
    print(f'[plot_random_bw] generated_run_dir_plots={len(run_dir_plots)}')
    if skipped:
        print(f'[plot_random_bw] skipped_run_dir_plots={len(skipped)}')
        print(f'[plot_random_bw] first_skip={skipped[0][0]}: {skipped[0][1]}')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Plot every flow's receiver goodput over time for one run.

Reads the per-flow ``iperf_receiver_flow{N}.json`` captures left in a run
directory and draws each flow's goodput (Mbps) on a shared absolute time axis
with the join time marked.  The per-flow state panels (throughput/CWND/RTT)
are rendered separately by the orchestrator-style episode plotter
(``runtime.render_state_plots``) into the same run directory.

Usage:
    ./venv_training/bin/python -m benchmarks_paper.plot_run <run_dir> [--out FILE]
"""

import argparse
import csv
import glob
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks_paper import paper_style as style
import matplotlib
import matplotlib.pyplot as plt


def _goodput_series(receiver_json: str, offset_s: float = 0.0) -> tuple:
    """Return (t_mid_s, mbps) from an iperf3 receiver JSON's intervals.

    iperf3 interval timestamps are relative to each flow's own start, so
    ``offset_s`` (the flow's scheduled episode start) is added to place the
    series on the shared absolute episode timeline.
    """
    try:
        with open(receiver_json) as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        return [], []
    t, y = [], []
    for interval in data.get('intervals', []) or []:
        summary = interval.get('sum', {}) or {}
        start = summary.get('start')
        end = summary.get('end')
        bps = summary.get('bits_per_second')
        if start is None or end is None or bps is None:
            continue
        t.append(offset_s + (float(start) + float(end)) / 2.0)
        y.append(float(bps) / 1e6)
    return t, y


def _flow_starts(run_dir: str) -> dict:
    path = os.path.join(run_dir, 'flow_plan.csv')
    out = {}
    try:
        with open(path, newline='') as handle:
            for row in csv.DictReader(handle):
                out[int(float(row['flow']))] = float(row['start_s'])
    except (OSError, KeyError, ValueError):
        pass
    return out


def plot_run(run_dir: str, out: str = None) -> int:
    """Render the combined-flows goodput figure for one run directory.

    Styling is applied inside an ``rc_context`` so the SciencePlots/LaTeX
    rcParams cannot leak into other plotters running in the same process
    (the orchestrator-style episode plots use plain matplotlib labels).
    """
    with matplotlib.rc_context():
        return _plot_run_styled(run_dir, out)


def _plot_run_styled(run_dir: str, out: str = None) -> int:
    run_dir = os.path.abspath(run_dir)
    receivers = sorted(glob.glob(os.path.join(run_dir, 'iperf_receiver_flow*.json')))
    if not receivers:
        print(f'[plot_run] no iperf_receiver_flow*.json in {run_dir}')
        return 1

    style.use_science()
    fig, ax = plt.subplots(figsize=(6.5, 2.6))
    starts = _flow_starts(run_dir)

    for index, receiver in enumerate(receivers):
        flow = int(''.join(ch for ch in os.path.basename(receiver)
                           if ch.isdigit()) or index + 1)
        color = style.PALETTE[(flow - 1) % len(style.PALETTE)]
        offset = float(starts.get(flow, 0.0))
        t, y = _goodput_series(receiver, offset_s=offset)
        if t:
            ax.plot(t, y, color=color, linewidth=1.0,
                    label=f'flow {flow} goodput')
        if flow in starts:
            ax.axvline(starts[flow], color=color, linestyle=':', linewidth=0.6,
                       alpha=0.6)

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Goodput (Mbps)')
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0)
    ax.set_title(_latexify(run_dir), fontsize=7)
    ax.legend(loc='upper right', fontsize=6, frameon=False)

    out = out or os.path.join(run_dir, 'run_flows')
    style.save(fig, Path(out).with_suffix(''))
    print(f'[plot_run] wrote {Path(out).with_suffix(".png")}')
    return 0


def _latexify(run_dir: str) -> str:
    tail = os.sep.join(run_dir.rstrip(os.sep).split(os.sep)[-4:])
    return tail.replace('_', r'\_')


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('run_dir')
    parser.add_argument('--out', default=None)
    args = parser.parse_args(argv)
    return plot_run(args.run_dir, args.out)


if __name__ == '__main__':
    raise SystemExit(main())

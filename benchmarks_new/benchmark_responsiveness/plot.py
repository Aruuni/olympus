#!/usr/bin/env python3
import argparse, json, os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
os.environ.setdefault('MPLCONFIGDIR', f'/tmp/matplotlib-{os.getuid()}')
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from benchmarks_new.plot_data import load_benchmark, number, output_root, run_label, state_rows

def ecdf(ax, values, label, **style):
    values = np.sort(np.asarray(values, float)[np.isfinite(values)])
    if values.size:
        ax.plot(values, np.arange(1, values.size + 1) / values.size * 100,
                label=label, linewidth=1.5, **style)

def scheduled(root):
    rows = []
    for path in root.glob('*/episodes/link_context_ep*.json'):
        data = json.loads(path.read_text()); base = data.get('base') or {}
        bws, rtts = [number(base, 'bw_mbps')], [number(base, 'rtt_us') / 1000]
        for event in data.get('events') or []:
            bws.append(number(event, 'bw')); rtts.append(number(event, 'delay'))
        rows.append((np.nanmean(bws), np.nanmean(rtts)))
    return rows

def main(argv=None):
    parser = argparse.ArgumentParser(); parser.add_argument('--config', default=str(Path(__file__).with_name('config.yaml'))); parser.add_argument('--output'); parser.add_argument('--debug', action='store_true')
    args = parser.parse_args(argv); config = Path(args.config).resolve()
    manifest, _ = load_benchmark(config, debug=args.debug); root = output_root(config, manifest)
    output = Path(args.output) if args.output else root / 'benchmark_summary.pdf'; output.parent.mkdir(parents=True, exist_ok=True)
    rows = state_rows(root)
    if not rows: print(f'[responsiveness_plot] no state logs beneath {root}'); return 1
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.8), constrained_layout=True)
    for run in sorted({row['run'] for row in rows}):
        subset = [row for row in rows if row['run'] == run]
        ecdf(axes[0], [row['throughput'] for row in subset], run_label(run)); ecdf(axes[1], [row['srtt'] for row in subset], run_label(run))
    references = scheduled(root); ecdf(axes[0], [v[0] for v in references], 'Scheduled BW', color='black'); ecdf(axes[1], [v[1] for v in references], 'Base RTT', color='black')
    for ax, title, label in zip(axes, ('Goodput CDF', 'SRTT CDF'), ('Average Goodput (Mbps)', 'Average SRTT (ms)')):
        ax.set(title=title, xlabel=label, ylabel='CDF (%)', ylim=(0, 100)); ax.grid(alpha=.25); ax.legend(frameon=False)
    fig.savefig(output); plt.close(fig); print(f'[responsiveness_plot] wrote {output}'); return 0
if __name__ == '__main__': raise SystemExit(main())

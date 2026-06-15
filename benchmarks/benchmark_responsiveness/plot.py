#!/usr/bin/env python3
"""Regenerate responsiveness plots from benchmark/data approach folders."""

import argparse
import os
import sys

os.environ.setdefault('MPLCONFIGDIR', os.path.join('/tmp', f'matplotlib-{os.getuid()}'))
os.makedirs(os.environ['MPLCONFIGDIR'], exist_ok=True)


_HERE = os.path.dirname(os.path.abspath(__file__))
_BENCHMARKS = os.path.dirname(_HERE)
_ROOT = os.path.dirname(_BENCHMARKS)
_DEFAULT_CONFIG = os.path.join(_HERE, 'config.yaml')
sys.path.insert(0, _ROOT)

from benchmarks.benchmark_responsiveness.responsiveness import (
    METRIC_FIELDS,
    _load_benchmark_config,
    _approach_label_map,
    _collect_approach_rows_with_labels,
    _enrich_kernel_rtt_rows,
    _plot_individual_runs,
    _plot_summary,
    _restore_sudo_user_ownership,
    _write_rows,
    _write_summary,
)


def main():
    ap = argparse.ArgumentParser(
        description='Plot every completed benchmark data folder under benchmark/data.')
    ap.add_argument('--config', default=_DEFAULT_CONFIG,
                    help='benchmark config yaml used for plot labels and default data dir')
    ap.add_argument('--data-dir', default=None,
                    help='benchmark data directory containing approach subfolders')
    ap.add_argument('--output', default=None,
                    help='output PDF; default is <data-dir>/responsiveness_summary.pdf')
    ap.add_argument('--skip-individual', action='store_true',
                    help='skip regenerating baseline protocol run plots')
    args = ap.parse_args()

    bench_cfg = {}
    if args.config:
        bench_cfg = _load_benchmark_config(os.path.abspath(args.config))

    data_dir_raw = args.data_dir
    if data_dir_raw is None:
        data_dir_raw = bench_cfg.get('output_root', 'data')
        if not os.path.isabs(str(data_dir_raw)):
            data_dir_raw = os.path.join(os.path.dirname(os.path.abspath(args.config)),
                                        str(data_dir_raw))

    data_dir = os.path.abspath(data_dir_raw)
    metrics_csv = os.path.join(data_dir, 'metrics.csv')
    summary_json = os.path.join(data_dir, 'summary.json')
    output_pdf = os.path.abspath(args.output or os.path.join(
        data_dir, 'responsiveness_summary.pdf'))

    rows = _enrich_kernel_rtt_rows(
        _collect_approach_rows_with_labels(data_dir, _approach_label_map(bench_cfg)))
    if not rows:
        raise SystemExit(f'[bench_plot] no approach metrics found under {data_dir}')

    if not args.skip_individual:
        plotted, skipped, failed = _plot_individual_runs(rows)
        print(f'[bench_plot] baseline_plots={plotted} skipped={skipped} failed={failed}',
              flush=True)

    _write_rows(metrics_csv, METRIC_FIELDS, rows)
    _write_summary(metrics_csv, summary_json)
    _plot_summary(metrics_csv, output_pdf)
    _restore_sudo_user_ownership(data_dir)
    print(f'[bench_plot] approaches={len({r["approach"] for r in rows})}', flush=True)
    print(f'[bench_plot] rows={len(rows)}', flush=True)
    print(f'[bench_plot] metrics={metrics_csv}', flush=True)
    print(f'[bench_plot] summary={summary_json}', flush=True)
    print(f'[bench_plot] plot={output_pdf}', flush=True)


if __name__ == '__main__':
    main()

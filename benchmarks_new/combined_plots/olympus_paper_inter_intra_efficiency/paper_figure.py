#!/usr/bin/env python3
"""Render the new intra/inter/efficiency benchmarks as one paper figure."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks_new.combined_plots.olympus_paper_inter_intra_efficiency.paper_efficiency import (
    efficiency_groups,
    scenario_settings as efficiency_settings,
)
from benchmarks_new.combined_plots.olympus_paper_inter_intra_efficiency.paper_fairness import (
    metric_groups,
    scenario_settings as fairness_settings,
)
from benchmarks_new.combined_plots.olympus_paper_inter_intra_efficiency.paper_plotting import (
    add_run_legend,
    draw_efficiency,
    draw_fairness,
    ordered_runs,
    save_figure,
    use_paper_style,
)
from benchmarks_new.plot_data import load_benchmark, output_root
import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
BENCHMARKS_DIR = HERE.parents[1]
DEFAULT_CONFIGS = {
    'intra': (
        BENCHMARKS_DIR / 'benchmark_PAPER_intra_rtt_fairness' / 'config.yaml'),
    'inter': (
        BENCHMARKS_DIR / 'benchmark_PAPER_inter_rtt_fairness' / 'config.yaml'),
    'efficiency': (
        BENCHMARKS_DIR / 'benchmark_paper_efficiency' / 'config.yaml'),
}


def _fairness_data(config, debug):
    manifest, overlay = load_benchmark(config, debug=debug)
    sweep = fairness_settings(config, manifest, overlay)
    return metric_groups(
        output_root(config, manifest),
        duration_s=float(sweep.get('duration', 200)),
        score_window_s=float(sweep.get('score_window_s', 100)),
    )


def _efficiency_data(config, debug):
    manifest, overlay = load_benchmark(config, debug=debug)
    sweep = efficiency_settings(config, manifest, overlay)
    return efficiency_groups(output_root(config, manifest), sweep)


def build_figure(configs, output, debug=False):
    intra = _fairness_data(configs['intra'], debug)
    inter = _fairness_data(configs['inter'], debug)
    efficiency = _efficiency_data(configs['efficiency'], debug)
    if not intra or not inter or not efficiency:
        missing = [
            name for name, groups in (
                ('intra', intra), ('inter', inter), ('efficiency', efficiency))
            if not groups
        ]
        raise ValueError(
            f'no benchmark episode data found for: {", ".join(missing)}')

    all_groups = {}
    all_groups.update(intra)
    all_groups.update(inter)
    all_groups.update(efficiency)
    runs = ordered_runs(all_groups)

    use_paper_style()
    fig, axes = plt.subplots(1, 3, figsize=(8.592, 1.776))
    draw_fairness(axes[0], intra, ordered_runs(intra))
    draw_fairness(axes[1], inter, ordered_runs(inter))
    draw_efficiency(
        axes[2],
        efficiency,
        ordered_runs(efficiency),
        connect_environments=True,
        ylabel='Norm. Thr.',
    )
    add_run_legend(fig, runs, bbox_y=0.98)
    fig.subplots_adjust(
        left=0.065,
        right=0.995,
        top=0.79,
        bottom=0.24,
        wspace=0.38,
    )
    save_figure(fig, output, png=True)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='Render the three new paper benchmarks as one figure.')
    parser.add_argument('--intra-config', type=Path,
                        default=DEFAULT_CONFIGS['intra'])
    parser.add_argument('--inter-config', type=Path,
                        default=DEFAULT_CONFIGS['inter'])
    parser.add_argument('--efficiency-config', type=Path,
                        default=DEFAULT_CONFIGS['efficiency'])
    parser.add_argument('--output', type=Path,
                        default=HERE / 'paper_figure.pdf')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args(argv)

    build_figure(
        {
            'intra': args.intra_config.resolve(),
            'inter': args.inter_config.resolve(),
            'efficiency': args.efficiency_config.resolve(),
        },
        args.output.resolve(),
        debug=args.debug,
    )
    print(f'[paper_figure] wrote {args.output.resolve()}')
    print(f'[paper_figure] wrote {args.output.resolve().with_suffix(".png")}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

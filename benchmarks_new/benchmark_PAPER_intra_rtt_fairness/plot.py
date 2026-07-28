#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
os.environ.setdefault('MPLCONFIGDIR', f'/tmp/matplotlib-{os.getuid()}')

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from benchmarks_new.combined_plots.olympus_paper_inter_intra_efficiency.paper_fairness import metric_groups, scenario_settings
from benchmarks_new.combined_plots.olympus_paper_inter_intra_efficiency.paper_plotting import (
    add_run_legend,
    draw_fairness,
    ordered_runs,
    save_figure,
    use_paper_style,
)
from benchmarks_new.plot_data import (
    environment_output,
    load_benchmark,
    output_root,
    run_environment,
)


def _write_plot(groups, runs, output, metric, ylabel, ylim):
    fig, axis = plt.subplots(figsize=(4, 1.2))
    if not draw_fairness(
            axis, groups, runs, metric=metric, ylabel=ylabel, ylim=ylim):
        plt.close(fig)
        return False
    add_run_legend(fig, runs, bbox_y=1.10)
    save_figure(fig, output)
    print(f'[paper_intra_plot] wrote {output}')
    return True


def _delay_output(output, environment=None):
    output = Path(output)
    delay = output.with_name(f'{output.stem}_delay{output.suffix}')
    return environment_output(delay, environment) if environment else delay


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config', default=str(Path(__file__).with_name('config.yaml')))
    parser.add_argument('--output')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args(argv)

    config = Path(args.config).resolve()
    manifest, overlay = load_benchmark(config, debug=args.debug)
    sweep = scenario_settings(config, manifest, overlay)
    root = output_root(config, manifest)
    output = Path(args.output) if args.output else root / 'benchmark_summary.pdf'
    output.parent.mkdir(parents=True, exist_ok=True)
    groups = metric_groups(
        root,
        duration_s=float(sweep.get('duration', 200)),
        score_window_s=float(sweep.get('score_window_s', 100)),
    )
    if not groups:
        print(f'[paper_intra_plot] no per-flow traces beneath {root}')
        return 1

    use_paper_style()
    runs = ordered_runs(groups)
    _write_plot(
        groups, runs, output,
        metric='goodput_ratio', ylabel='Goodput Ratio', ylim=(-0.1, 1.1))
    _write_plot(
        groups, runs, _delay_output(output),
        metric='delay_ratio', ylabel='Delay Ratio', ylim=None)
    for environment in ('mininet', 'raynet'):
        environment_runs = [
            run for run in runs if run_environment(run) == environment]
        if environment_runs:
            _write_plot(
                groups,
                environment_runs,
                environment_output(output, environment),
                metric='goodput_ratio',
                ylabel='Goodput Ratio',
                ylim=(-0.1, 1.1),
            )
            _write_plot(
                groups,
                environment_runs,
                _delay_output(output, environment),
                metric='delay_ratio',
                ylabel='Delay Ratio',
                ylim=None,
            )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

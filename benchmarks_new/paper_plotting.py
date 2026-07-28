"""Shared paper-style drawing primitives for the new benchmark suites."""

import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', f'/tmp/matplotlib-{os.getuid()}')

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
from matplotlib.ticker import AutoMinorLocator, FormatStrFormatter, ScalarFormatter
import matplotlib.transforms as mtransforms
import numpy as np

from benchmarks_new.paper_efficiency import group_points
from benchmarks_new.paper_fairness import series
from benchmarks_new.plot_data import run_environment, run_label


LINEWIDTH = 0.30
ELINEWIDTH = 0.75
CAPTHICK = ELINEWIDTH
CAPSIZE = 2

PROTOCOL_ORDER = ('astraea', 'orca', 'dreamer')
PROTOCOL_STYLES = {
    'astraea': {
        'marker': '^',
        'raynet': 'tab:blue',
        'mininet': 'tab:green',
        'connector': 'tab:blue',
    },
    'orca': {
        'marker': 'D',
        'raynet': 'tab:orange',
        'mininet': 'tab:purple',
        'connector': 'tab:orange',
    },
    'dreamer': {
        'marker': 'P',
        'raynet': 'tab:red',
        'mininet': 'tab:brown',
        'connector': 'tab:red',
    },
}
FALLBACK_COLORS = ('#0C5DA5', '#00B945', '#FF9500', '#845B97', '#FF2C01')
FALLBACK_MARKERS = ('o', 's', '^', 'D', 'x')
RTT_MARKERS = {
    10.0: 'v',
    100.0: 's',
}
FALLBACK_RTT_MARKERS = ('o', 'D', '^', '*')


def use_paper_style():
    """Approximate the legacy SciencePlots style without adding a dependency."""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Computer Modern Roman', 'DejaVu Serif'],
        'mathtext.fontset': 'cm',
        'font.size': 7.5,
        'axes.labelsize': 7.5,
        'axes.titlesize': 7.5,
        'axes.linewidth': 0.5,
        'axes.grid': False,
        'legend.fontsize': 7,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.major.width': 0.5,
        'ytick.major.width': 0.5,
        'xtick.minor.width': 0.4,
        'ytick.minor.width': 0.4,
        'lines.linewidth': LINEWIDTH,
        'savefig.dpi': 1080,
    })


def protocol(run):
    return str(run).split('__', 1)[0].lower()


def ordered_runs(groups):
    """Return stable protocol/backend ordering for either paper data shape."""
    runs = {key[0] for key in groups}
    protocol_rank = {name: index for index, name in enumerate(PROTOCOL_ORDER)}
    environment_rank = {'raynet': 0, 'mininet': 1}
    return sorted(
        runs,
        key=lambda run: (
            protocol_rank.get(protocol(run), len(protocol_rank)),
            protocol(run),
            environment_rank.get(run_environment(run), 2),
            str(run),
        ),
    )


def run_style(run, index=0):
    style = PROTOCOL_STYLES.get(protocol(run), {})
    environment = run_environment(run)
    return (
        style.get(environment, FALLBACK_COLORS[index % len(FALLBACK_COLORS)]),
        style.get('marker', FALLBACK_MARKERS[index % len(FALLBACK_MARKERS)]),
    )


def paper_run_label(run):
    label = run_label(run)
    environment = run_environment(run)
    if not environment:
        return label
    base = label.rsplit(' — ', 1)[0]
    environment_label = {
        'mininet': 'Mininet',
        'raynet': 'RayNet',
    }.get(environment, environment.title())
    return f'{base} ({environment_label})'


def _finish_axis(axis):
    axis.grid(False)
    axis.xaxis.set_minor_locator(AutoMinorLocator(5))
    axis.yaxis.set_minor_locator(AutoMinorLocator(5))
    axis.tick_params(axis='both', which='both', direction='in')


def draw_fairness(
        axis, groups, runs, metric='goodput_ratio',
        ylabel='Goodput Ratio', ylim=(-0.1, 1.1)):
    """Draw one legacy-style RTT fairness panel."""
    drew = False
    for index, run in enumerate(runs):
        rtt_ms, mean, std = series(groups, run, metric)
        if not rtt_ms.size:
            continue
        color, marker = run_style(run, index)
        _, caps, bars = axis.errorbar(
            rtt_ms,
            mean,
            yerr=std,
            marker=marker,
            markersize=4.0,
            markeredgewidth=0.8,
            linewidth=LINEWIDTH,
            elinewidth=ELINEWIDTH,
            capsize=CAPSIZE,
            capthick=CAPTHICK,
            color=color,
            label=paper_run_label(run),
        )
        for artist in (*caps, *bars):
            artist.set_alpha(0.5)
        drew = True

    axis.set(yscale='linear', xlabel='RTT (ms)', ylabel=ylabel)
    if ylim is not None:
        axis.set_ylim(ylim)
    axis.xaxis.set_major_formatter(ScalarFormatter())
    axis.yaxis.set_major_formatter(ScalarFormatter())
    _finish_axis(axis)
    return drew


def _confidence_ellipse(x, y, axis, color):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 2 or x.size != y.size:
        return None
    covariance = np.cov(x, y)
    if not np.all(np.isfinite(covariance)):
        return None
    x_var, y_var = float(covariance[0, 0]), float(covariance[1, 1])
    if x_var <= 0 or y_var <= 0:
        return None
    pearson = float(np.clip(
        covariance[0, 1] / np.sqrt(x_var * y_var), -1.0, 1.0))
    ellipse = Ellipse(
        (0, 0),
        width=2 * np.sqrt(1 + pearson),
        height=2 * np.sqrt(1 - pearson),
        facecolor=color,
        edgecolor='none',
        alpha=0.25,
    )
    transform = (
        mtransforms.Affine2D()
        .rotate_deg(45)
        .scale(np.sqrt(x_var), np.sqrt(y_var))
        .translate(float(np.mean(x)), float(np.mean(y)))
    )
    ellipse.set_transform(transform + axis.transData)
    return axis.add_patch(ellipse)


def _rtt_markers(rtts):
    return {
        rtt: RTT_MARKERS.get(
            float(rtt),
            FALLBACK_RTT_MARKERS[index % len(FALLBACK_RTT_MARKERS)],
        )
        for index, rtt in enumerate(rtts)
    }


def draw_efficiency(
        axis, groups, runs, connect_environments=True,
        ylabel='Norm. Throughput'):
    """Draw one legacy-style normalized delay/throughput panel."""
    selected = {key: value for key, value in groups.items() if key[0] in runs}
    rtts = sorted({key[2] for key in selected})
    markers = _rtt_markers(rtts)
    mean_points = {}
    drew = False

    for index, ((run, bw_mbps, rtt_ms), group) in enumerate(
            sorted(selected.items())):
        points = group_points(group)
        if not points.size:
            continue
        color, _ = run_style(run, index)
        delay, throughput = points[:, 0], points[:, 1]
        mean_point = (float(np.mean(delay)), float(np.mean(throughput)))
        mean_points[(run, bw_mbps, rtt_ms)] = mean_point
        _confidence_ellipse(delay, throughput, axis, color)
        axis.scatter(
            [mean_point[0]],
            [mean_point[1]],
            s=22,
            edgecolors=color,
            facecolors='none',
            marker=markers[rtt_ms],
            linewidths=0.8,
            zorder=3,
        )
        drew = True

    if connect_environments:
        paired = {}
        for (run, bw_mbps, rtt_ms), point in mean_points.items():
            key = (protocol(run), bw_mbps, rtt_ms)
            paired.setdefault(key, {})[run_environment(run)] = point
        for (name, _, _), environments in paired.items():
            if 'mininet' not in environments or 'raynet' not in environments:
                continue
            mininet = environments['mininet']
            raynet = environments['raynet']
            axis.plot(
                [mininet[0], raynet[0]],
                [mininet[1], raynet[1]],
                color=PROTOCOL_STYLES.get(name, {}).get(
                    'connector', '#777777'),
                linewidth=0.7,
                alpha=0.8,
                zorder=2,
            )

    axis.set(
        xlabel='Norm. Delay',
        ylabel=ylabel,
        ylim=(0, 1.05),
    )
    axis.invert_xaxis()
    axis.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    axis.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    _finish_axis(axis)

    handles = [
        Line2D(
            [], [], color='black', linestyle='None', marker=markers[rtt],
            markerfacecolor='none', markersize=5, markeredgewidth=0.8,
        )
        for rtt in rtts
    ]
    if handles:
        axis.legend(
            handles,
            [f'{rtt:g} ms' for rtt in rtts],
            loc='lower right',
            frameon=False,
            fontsize=6,
            handlelength=0.8,
            handletextpad=0.6,
            labelspacing=0.2,
            title='RTT',
            title_fontsize=6,
        )
    return drew


def add_run_legend(fig, runs, bbox_y=1.10, ncol=None):
    handles = []
    labels = []
    for index, run in enumerate(runs):
        color, marker = run_style(run, index)
        handles.append(Line2D(
            [], [],
            color=color,
            marker=marker,
            linewidth=LINEWIDTH,
            markersize=4,
            markeredgewidth=0.8,
        ))
        labels.append(paper_run_label(run))
    if handles:
        fig.legend(
            handles,
            labels,
            ncol=ncol or len(handles),
            loc='upper center',
            bbox_to_anchor=(0.5, bbox_y),
            frameon=False,
            fontsize=7,
            columnspacing=0.8,
            handlelength=1.8,
            handletextpad=0.5,
        )


def save_figure(fig, output, png=False):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix:
        fig.savefig(output, dpi=1080, bbox_inches='tight')
        if png:
            fig.savefig(output.with_suffix('.png'), dpi=300, bbox_inches='tight')
    else:
        fig.savefig(output.with_suffix('.pdf'), dpi=1080, bbox_inches='tight')
        if png:
            fig.savefig(output.with_suffix('.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)

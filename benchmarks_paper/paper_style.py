#!/usr/bin/env python3
"""Shared IFIP-2026 figure styling for the inter-/intra-RTT fairness plots.

This mirrors the ``core/plotting.py`` header from Mihai Mazilu's mininettestbed
so the Olympus reproductions look byte-for-byte like the paper figures: the
SciencePlots ``science`` style with LaTeX text, the same ``plot_points``
error-bar helper, and the paper's per-protocol colour/marker palette.
"""

import math
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', f'/tmp/matplotlib-{os.getuid()}')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import scienceplots  # noqa: F401,E402  (registers the 'science' style)


# --- Paper palette (from core/plotting.py) -------------------------------
LINEWIDTH = 0.30
ELINEWIDTH = 0.75
CAPTHICK = ELINEWIDTH
CAPSIZE = 2

COLORS_LEO = {
    'cubic': '#0C5DA5', 'orca': '#00B945', 'bbr3': '#FF9500',
    'sage': '#FF2C01', 'vivace': '#845B97', 'astraea': '#00B945',
    'vivace-uspace': '#845B97', 'bbr1': '#964B00', 'satcp': '#000000',
    'dreamer': '#845B97',
}
MARKERS_LEO = {
    'cubic': 'x', 'orca': '+', 'bbr3': '.', 'sage': '*', 'vivace': '4',
    'astraea': '2', 'vivace-uspace': '_', 'bbr1': '1', 'dreamer': 'D',
}

# Fallback cycle for approaches that do not name a paper protocol (e.g. two
# distinct Orca variants that must be told apart on the same axes).
PALETTE = ['#00B945', '#0C5DA5', '#FF9500', '#845B97', '#FF2C01', '#686868']
MARKERS = ['o', 's', '^', 'D', 'x', '+']


def use_science() -> None:
    """Activate the paper's SciencePlots style with LaTeX text rendering."""
    plt.style.use('science')
    plt.rcParams.update({
        'text.usetex': True,
        'font.size': 7.5,
        'legend.fontsize': 7,
    })


def _latex_escape(text: str) -> str:
    return str(text).replace('&', r'\&').replace('%', r'\%').replace('_', r'\_')


def style_for(approach_row: dict, index: int) -> tuple:
    """Return (color, marker, legend_label) for one configured approach.

    An approach may pin the paper look with ``protocol: orca`` (looks up the
    exact paper colour/marker) or override ``color``/``marker`` explicitly;
    otherwise it cycles the fallback palette.  The legend uses ``plot_label``.
    """
    protocol = str(approach_row.get('protocol', '')).strip().lower()
    color = approach_row.get('color') or COLORS_LEO.get(
        protocol, PALETTE[index % len(PALETTE)])
    marker = approach_row.get('marker') or MARKERS_LEO.get(
        protocol, MARKERS[index % len(MARKERS)])
    label = (approach_row.get('plot_label') or approach_row.get('data_folder')
             or approach_row.get('name') or protocol or f'approach {index}')
    return color, marker, _latex_escape(label)


def plot_points(ax, xvals, yvals, yerr, marker, color, label) -> None:
    """Error-bar series identical to the paper's ``core.plotting.plot_points``."""
    if not len(xvals):
        return
    _, caps, bars = ax.errorbar(
        xvals, yvals, yerr=yerr,
        marker=marker, linewidth=LINEWIDTH, elinewidth=ELINEWIDTH,
        capsize=CAPSIZE, capthick=CAPTHICK, color=color, label=label)
    for bar in bars:
        bar.set_alpha(0.5)
    for cap in caps:
        cap.set_alpha(0.5)


def save(fig, stem: Path) -> None:
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix('.pdf'), dpi=1080, bbox_inches='tight')
    fig.savefig(stem.with_suffix('.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)


def finite(value, default=math.nan) -> float:
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default

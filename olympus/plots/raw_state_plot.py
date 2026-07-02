"""Plot schema-free raw worker observations captured as JSONL."""

import json
import math
import os
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

_META_KEYS = {
    '_agent_id', '_backend', '_cport', '_flow_id', '_pid', '_wall_t_s',
    'collection_participants', 'group_step', 'step_id',
}


def _numeric(value):
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _flow_label(path, row):
    for key in ('_flow_id', '_agent_id'):
        value = row.get(key)
        if value not in (None, ''):
            return f'flow {value}'
    match = re.search(r'_flow([^./]+)', os.path.basename(path))
    if match:
        return f'flow {match.group(1)}'
    return os.path.basename(path)


def _load(path):
    rows = []
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError:
        return []
    return rows


def _x_value(row):
    value = _numeric(row.get('time_s'))
    if value is not None:
        return value
    return _numeric(row.get('_wall_t_s'))


def plot(raw_logs, output, title='Raw Observations'):
    series = []
    keys = set()
    for path in raw_logs:
        rows = _load(path)
        if not rows:
            continue
        label = _flow_label(path, rows[0])
        series.append((label, rows))
        for row in rows:
            for key, value in row.items():
                if key in _META_KEYS or key.startswith('_') or key == 'time_s':
                    continue
                if _numeric(value) is not None:
                    keys.add(key)
    if not series or not keys:
        return False

    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    with PdfPages(output) as pdf:
        for key in sorted(keys):
            fig, ax = plt.subplots(figsize=(12, 5))
            fig.suptitle(f'{title} - {key}', fontsize=12, fontweight='bold')
            plotted = False
            for label, rows in series:
                points = []
                for row in rows:
                    x = _x_value(row)
                    y = _numeric(row.get(key))
                    if x is None or y is None:
                        continue
                    points.append((x, y))
                if not points:
                    continue
                points.sort(key=lambda item: item[0])
                xs, ys = zip(*points)
                ax.plot(xs, ys, linewidth=1.0, alpha=0.85, label=label)
                plotted = True
            if not plotted:
                plt.close(fig)
                continue
            ax.set_xlabel('sim time (s)' if any('time_s' in r for _, rows in series for r in rows[:1]) else 'time (s)')
            ax.set_ylabel(key)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8, framealpha=0.85)
            pdf.savefig(fig, dpi=120, bbox_inches='tight')
            plt.close(fig)
    return True


def raw_state_plot_path(episode_plot_path):
    root, ext = os.path.splitext(str(episode_plot_path))
    return f'{root}_raw_state{ext or ".pdf"}'

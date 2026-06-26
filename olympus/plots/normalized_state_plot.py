"""Plot normalized state columns written by rollout workers."""

import csv
import os
import re

import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np


_COLORS = [
    '#4878cf', '#f28e2c', '#59a14f', '#e15759',
    '#b07aa1', '#76b7b2', '#ff9da7', '#9c755f',
    '#bab0ac', '#edc948', '#d37295', '#54a24b',
]


def _state_columns(fieldnames):
    pairs = []
    for name in fieldnames or []:
        match = re.fullmatch(r's(\d+)', name)
        if match:
            pairs.append((int(match.group(1)), name))
    return [name for _, name in sorted(pairs)]


def _load(path, trim_tail_s=5.0):
    t_s = []
    values = {}
    state_cols = []
    try:
        with open(path, newline='') as handle:
            reader = csv.DictReader(handle)
            state_cols = _state_columns(reader.fieldnames)
            values = {name: [] for name in state_cols}
            for row in reader:
                try:
                    t_s.append(float(row['t_s']))
                    for name in state_cols:
                        values[name].append(float(row[name]))
                except (KeyError, TypeError, ValueError):
                    if t_s:
                        t_s.pop()
                    for name in state_cols:
                        if len(values[name]) > len(t_s):
                            values[name].pop()
    except FileNotFoundError:
        return None

    if not t_s or not state_cols:
        return None

    data = {'t_s': np.asarray(t_s, dtype=np.float32)}
    for name in state_cols:
        data[name] = np.asarray(values[name], dtype=np.float32)

    if trim_tail_s and len(data['t_s']) > 1:
        mask = data['t_s'] <= data['t_s'][-1] - float(trim_tail_s)
        if mask.any():
            data = {key: value[mask] for key, value in data.items()}
    return data


def _trace_label(path, fallback):
    name = os.path.basename(path)
    match = re.search(r'_a(\d+)(?:\.[^.]+)?$', name)
    if match:
        return f'agent {match.group(1)}'
    match = re.search(r'_flow(\d+)(?:\.[^.]+)?$', name)
    if match:
        return f'flow {match.group(1)}'
    return fallback


def _state_plot_path(episode_plot_path):
    root, ext = os.path.splitext(str(episode_plot_path))
    return f'{root}_normalized_state{ext or ".pdf"}'


def plot(state_logs, output, title=None, trim_tail_s=5.0):
    traces = []
    for index, path in enumerate(state_logs or []):
        data = _load(path, trim_tail_s=trim_tail_s)
        if data is None:
            continue
        traces.append({
            'label': _trace_label(path, f'trace {index}'),
            'data': data,
        })

    if not traces:
        print(f'[state_plot] no normalized state data for: {output}', flush=True)
        return None

    state_cols = [
        name for name in traces[0]['data'].keys()
        if re.fullmatch(r's\d+', name)
    ]
    state_cols = _state_columns(state_cols)
    if not state_cols:
        print(f'[state_plot] no s* columns for: {output}', flush=True)
        return None

    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    with PdfPages(output) as pdf:
        for col_index, col in enumerate(state_cols):
            fig, ax = plt.subplots(figsize=(13, 5))
            if title:
                fig.suptitle(f'{title}  {col}', fontsize=12, fontweight='bold')
            else:
                fig.suptitle(f'Normalized state {col}', fontsize=12, fontweight='bold')

            for trace_index, trace in enumerate(traces):
                data = trace['data']
                if col not in data:
                    continue
                ax.plot(
                    data['t_s'],
                    data[col],
                    color=_COLORS[trace_index % len(_COLORS)],
                    linewidth=0.9,
                    alpha=0.85,
                    label=trace['label'],
                )

            ax.axhline(0.0, color='black', linewidth=0.5, alpha=0.35)
            ax.set_xlabel('Time (s)')
            ax.set_ylabel(col)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8, loc='best')
            plt.tight_layout(rect=[0, 0.02, 1, 0.94])
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

    print(f'[state_plot] saved -> {output}', flush=True)
    return output


__all__ = ['plot', '_state_plot_path']

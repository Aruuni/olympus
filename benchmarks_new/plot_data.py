"""Backend-neutral readers for Olympus benchmark episode artifacts."""

import csv
from pathlib import Path
import re

import numpy as np

from benchmarks_new.common import load_approaches, load_benchmark


def run_label(run):
    """Convert an eval run directory name into a configured plot label."""
    parts = str(run).split('__')
    approach = load_approaches().get(parts[0], {})
    label = approach.get('plot_label') or parts[0]
    return f'{label} — {parts[-1]}' if len(parts) > 1 else label


def run_environment(run):
    """Return the backend encoded in an evaluation run directory name."""
    parts = str(run).split('__')
    return parts[-1].lower() if len(parts) > 1 else None


def environment_output(output: Path, environment: str) -> Path:
    """Return the per-environment sibling of a combined plot path."""
    return output.with_name(
        f'{output.stem}_{environment.lower()}{output.suffix}')


def number(row, key):
    try:
        return float(row.get(key, ''))
    except (TypeError, ValueError):
        return np.nan


def output_root(config: Path, manifest: dict) -> Path:
    value = (manifest.get('defaults') or {}).get('output_root', 'data')
    path = Path(value)
    return path if path.is_absolute() else (config.parent / path).resolve()


def state_rows(root: Path):
    records = []
    for path in root.glob('*/episodes/*_state_ep*.csv'):
        with path.open(newline='') as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            continue
        times = np.asarray([number(row, 't_s') for row in rows])
        keep = np.isfinite(times)
        if keep.any():
            trimmed = times <= np.nanmax(times[keep]) - 5.0
            if trimmed.any():
                keep &= trimmed

        def positive_mean(key):
            values = np.asarray([number(row, key) for row in rows])
            selected = values[keep & np.isfinite(values) & (values > 0)]
            return float(np.mean(selected)) if selected.size else np.nan

        match = re.search(r'_ep(\d+)', path.stem)
        records.append({
            'run': path.parents[1].name,
            'flow': path.stem,
            'episode': int(match.group(1)) if match else None,
            'throughput': positive_mean('avg_thr_mbps'),
            'rtt': positive_mean('avg_urtt_ms'),
            'srtt': positive_mean('srtt_ms'),
        })
    return records


def return_rows(root: Path):
    rows = []
    for path in root.glob('*/episodes/episode_returns.csv'):
        with path.open(newline='') as handle:
            rows.extend(dict(row, run_dir=path.parents[1].name)
                        for row in csv.DictReader(handle))
    return rows


def episode_traces(root: Path):
    episodes = {}
    for path in root.glob('*/episodes/*_state_ep*.csv'):
        episode = re.search(r'_ep(\d+)', path.name)
        if not episode:
            continue
        flow = re.search(r'_(?:flow|a)(\d+)(?:\.|$)', path.name)
        with path.open(newline='') as handle:
            rows = list(csv.DictReader(handle))
        data = {
            key: np.asarray([number(row, key) for row in rows])
            for key in ('t_s', 'avg_thr_mbps', 'srtt_ms', 'avg_urtt_ms')
        }
        key = (path.parents[1].name, int(episode.group(1)))
        episodes.setdefault(key, []).append((int(flow.group(1)) if flow else 0, data))
    for traces in episodes.values():
        traces.sort(key=lambda item: item[0])
    return episodes


def aligned_throughput(traces, step_s=1.0):
    valid = []
    for flow, data in traces:
        t, throughput = data['t_s'], data['avg_thr_mbps']
        mask = np.isfinite(t) & np.isfinite(throughput) & (throughput >= 0)
        if mask.any():
            valid.append((flow, t[mask], throughput[mask]))
    if not valid:
        return None
    end = max(float(t[-1]) for _, t, _ in valid) - 5.0
    if end <= 0:
        return None
    grid = np.arange(0.0, end + 1e-9, step_s)
    matrix = np.full((len(valid), len(grid)), np.nan)
    for index, (_, t, throughput) in enumerate(valid):
        active = (grid >= t[0]) & (grid <= min(t[-1], end))
        matrix[index, active] = np.interp(grid[active], t, throughput)
    return grid, matrix


def metadata_index(rows):
    result = {}
    for row in rows:
        try:
            result[(row['run_dir'], int(row['episode']))] = row
        except (KeyError, TypeError, ValueError):
            pass
    return result

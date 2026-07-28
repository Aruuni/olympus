"""Shared metrics for the paper-style inter-/intra-RTT benchmarks."""

from pathlib import Path

import numpy as np

from benchmarks_new.common import load_scenario
from benchmarks_new.plot_data import (
    aligned_throughput,
    episode_traces,
    metadata_index,
    number,
    return_rows,
)


def scenario_settings(config: Path, manifest: dict, overlay: dict) -> dict:
    """Return plotting parameters from the benchmark's selected scenario."""
    scenario_value = (manifest.get('matrix') or {}).get('scenarios', [None])[0]
    if not scenario_value:
        return {}
    path = Path(scenario_value)
    path = path if path.is_absolute() else config.parent / path
    return load_scenario(path, overlay).get('sweep') or {}


def _score_mask(time_s, throughput, duration_s, score_window_s):
    active_count = np.isfinite(throughput).sum(axis=0)
    start = max(0.0, float(duration_s) - float(score_window_s))
    return ((time_s >= start) & (time_s < float(duration_s))
            & (active_count >= throughput.shape[0]))


def _episode_metrics(traces, duration_s, score_window_s, base_rtt_ms):
    """Return one score-window mean per metric for one episode."""
    aligned = aligned_throughput(traces)
    if aligned is None:
        return np.nan, np.nan
    time_s, throughput = aligned
    if throughput.shape[0] < 2:
        return np.nan, np.nan
    mask = _score_mask(
        time_s, throughput, duration_s=duration_s,
        score_window_s=score_window_s)

    ratios = []
    for index in np.flatnonzero(mask):
        values = throughput[:, index]
        values = values[np.isfinite(values)]
        if values.size >= 2 and float(np.max(values)) > 0:
            ratios.append(float(np.min(values)) / float(np.max(values)))

    delay_samples = []
    score_start = max(0.0, float(duration_s) - float(score_window_s))
    if np.isfinite(base_rtt_ms) and base_rtt_ms > 0:
        for _, data in traces:
            sample_time = np.asarray(data.get('t_s', []), dtype=float)
            srtt_ms = np.asarray(data.get('srtt_ms', []), dtype=float)
            keep = ((sample_time >= score_start)
                    & (sample_time < float(duration_s))
                    & np.isfinite(srtt_ms) & (srtt_ms > 0))
            delay_samples.extend(srtt_ms[keep] / float(base_rtt_ms))

    ratios = np.asarray(ratios, dtype=float)
    delay_samples = np.asarray(delay_samples, dtype=float)
    return (
        float(np.mean(ratios)) if ratios.size else np.nan,
        float(np.mean(delay_samples)) if delay_samples.size else np.nan,
    )


def metric_groups(root: Path, duration_s: float, score_window_s: float) -> dict:
    """Group episode-level means by ``(evaluation run, swept RTT)``.

    An evaluation run identifies the checkpoint, scenario, and backend, while
    each repeated episode contributes exactly one value to the distribution.
    This keeps error bars representative of run-to-run variation instead of
    treating correlated samples within an episode as independent repetitions.
    """
    metadata = metadata_index(return_rows(root))
    groups = {}
    for key, traces in episode_traces(root).items():
        meta = metadata.get(key, {})
        swept_rtt_ms = number(meta, 'delay')
        if not np.isfinite(swept_rtt_ms) or swept_rtt_ms <= 0:
            continue
        goodput_ratio, delay_ratio = _episode_metrics(
            traces,
            duration_s=duration_s,
            score_window_s=score_window_s,
            base_rtt_ms=swept_rtt_ms,
        )
        if not np.isfinite(goodput_ratio) and not np.isfinite(delay_ratio):
            continue
        repetition = number(meta, 'repetition')
        repetition = int(repetition) if np.isfinite(repetition) else int(key[1])
        group = groups.setdefault((key[0], swept_rtt_ms), {
            'goodput_ratio': {},
            'delay_ratio': {},
            '_episode': {},
        })
        # Evaluation output can retain artifacts when the same configured
        # repetition is rerun. Keep its newest episode instead of inflating N.
        if int(key[1]) < group['_episode'].get(repetition, -1):
            continue
        group['_episode'][repetition] = int(key[1])
        if np.isfinite(goodput_ratio):
            group['goodput_ratio'][repetition] = goodput_ratio
        else:
            group['goodput_ratio'].pop(repetition, None)
        if np.isfinite(delay_ratio):
            group['delay_ratio'][repetition] = delay_ratio
        else:
            group['delay_ratio'].pop(repetition, None)
    return groups


def series(groups: dict, run: str, metric: str):
    """Return RTT, mean episode score, and between-episode standard deviation."""
    rows = []
    for (group_run, rtt_ms), values in groups.items():
        if group_run != run:
            continue
        samples = np.asarray(
            list(values.get(metric, {}).values()), dtype=float)
        samples = samples[np.isfinite(samples)]
        if not samples.size:
            continue
        rows.append((
            float(rtt_ms),
            float(np.mean(samples)),
            float(np.std(samples, ddof=1)) if samples.size > 1 else np.nan,
        ))
    rows.sort()
    if not rows:
        return np.asarray([]), np.asarray([]), np.asarray([])
    return tuple(np.asarray(values, dtype=float) for values in zip(*rows))

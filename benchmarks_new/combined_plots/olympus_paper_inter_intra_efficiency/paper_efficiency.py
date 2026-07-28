"""Metrics for the paper Figure 9 efficiency benchmark."""

from pathlib import Path

import numpy as np

from benchmarks_new.common import load_scenario
from benchmarks_new.plot_data import (
    episode_traces,
    metadata_index,
    number,
    return_rows,
)


def scenario_settings(config: Path, manifest: dict, overlay: dict) -> dict:
    """Return plotting parameters from the selected efficiency scenario."""
    scenario_value = (manifest.get('matrix') or {}).get('scenarios', [None])[0]
    if not scenario_value:
        return {}
    path = Path(scenario_value)
    path = path if path.is_absolute() else config.parent / path
    return load_scenario(path, overlay).get('sweep') or {}


def _last_sample_at(query, sample_time, values):
    """Sample a state trace with the paper benchmark's step interpolation."""
    output = np.full(query.shape, np.nan, dtype=float)
    valid = np.isfinite(sample_time) & np.isfinite(values)
    sample_time, values = sample_time[valid], values[valid]
    if not sample_time.size:
        return output
    order = np.argsort(sample_time)
    sample_time, values = sample_time[order], values[order]
    indices = np.searchsorted(sample_time, query, side='right') - 1
    present = indices >= 0
    output[present] = values[indices[present]]
    return output


def episode_efficiency(
        traces, *, bw_mbps, base_rtt_ms, duration_s, starts,
        flow_duration_s, score_window_s):
    """Return one normalized-delay/throughput point for an episode.

    Throughput follows the paper pipeline: aggregate active-flow goodput is
    sampled once per second over the whole experiment and normalized by link
    capacity. Delay pools positive SRTT samples from each flow's own first
    ``score_window_s`` seconds and normalizes them by the configured base RTT.
    """
    if bw_mbps <= 0 or base_rtt_ms <= 0 or duration_s <= 0:
        return None
    by_flow = {int(flow): data for flow, data in traces}
    flow_count = len(starts)
    if flow_count == 0 or any(flow not in by_flow for flow in range(flow_count)):
        return None

    grid = np.arange(1.0, float(duration_s) + 1.0, 1.0)
    total = np.zeros(grid.shape, dtype=float)
    active_count = np.zeros(grid.shape, dtype=int)
    observed_count = np.zeros(grid.shape, dtype=int)
    delay_samples = []

    for flow, start in enumerate(starts):
        start = float(start)
        end = min(float(duration_s), start + float(flow_duration_s))
        data = by_flow[flow]
        sample_time = np.asarray(data.get('t_s', []), dtype=float)
        throughput = np.asarray(data.get('avg_thr_mbps', []), dtype=float)
        sampled = _last_sample_at(grid, sample_time, throughput)
        active = (grid >= start) & (grid < end)
        observed = active & np.isfinite(sampled)
        total += np.where(observed, sampled, 0.0)
        active_count += active.astype(int)
        observed_count += observed.astype(int)

        srtt_ms = np.asarray(data.get('srtt_ms', []), dtype=float)
        delay_end = min(end, start + float(score_window_s))
        keep = (
            np.isfinite(sample_time)
            & np.isfinite(srtt_ms)
            & (srtt_ms > 0)
            & (sample_time >= start)
            & (sample_time < delay_end)
        )
        delay_samples.extend(srtt_ms[keep] / float(base_rtt_ms))

    # Mininet's listener may attach shortly after a flow starts. Those
    # pre-attachment values are unknown, not zero throughput. Compare only
    # seconds for which every scheduled-active flow has an observation.
    complete = (active_count > 0) & (observed_count == active_count)
    throughput_samples = total[complete] / float(bw_mbps)
    throughput_samples = throughput_samples[np.isfinite(throughput_samples)]
    delay_samples = np.asarray(delay_samples, dtype=float)
    delay_samples = delay_samples[np.isfinite(delay_samples)]
    if not throughput_samples.size or not delay_samples.size:
        return None
    return (
        float(np.mean(delay_samples)),
        float(np.mean(throughput_samples)),
    )


def efficiency_groups(root: Path, sweep: dict) -> dict:
    """Group newest episode-level points by run, condition, and repetition."""
    duration_s = float(sweep.get('duration', 200))
    score_window_s = float(sweep.get('score_window_s', 100))
    flow_schedule = sweep.get('flow_schedule') or {}
    arrival = flow_schedule.get('arrival') or {}
    starts = [float(value) for value in arrival.get(
        'start_delays', [0, 25, 50, 75])]
    duration_spec = flow_schedule.get('duration') or {}
    flow_duration_s = float(duration_spec.get('fixed_s', 125))

    metadata = metadata_index(return_rows(root))
    groups = {}
    for key, traces in episode_traces(root).items():
        meta = metadata.get(key, {})
        bw_mbps = number(meta, 'bw')
        base_rtt_ms = number(meta, 'delay')
        if not (np.isfinite(bw_mbps) and bw_mbps > 0
                and np.isfinite(base_rtt_ms) and base_rtt_ms > 0):
            continue
        point = episode_efficiency(
            traces,
            bw_mbps=bw_mbps,
            base_rtt_ms=base_rtt_ms,
            duration_s=duration_s,
            starts=starts,
            flow_duration_s=flow_duration_s,
            score_window_s=score_window_s,
        )
        if point is None:
            continue
        repetition = number(meta, 'repetition')
        repetition = int(repetition) if np.isfinite(repetition) else int(key[1])
        group = groups.setdefault(
            (key[0], float(bw_mbps), float(base_rtt_ms)), {})
        previous = group.get(repetition)
        if previous is None or int(key[1]) >= previous[0]:
            group[repetition] = (int(key[1]), point)
    return groups


def group_points(group: dict):
    """Return one point for each declared repetition in a condition."""
    return np.asarray(
        [entry[1] for _, entry in sorted(group.items())], dtype=float)

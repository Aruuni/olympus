"""Backend-neutral per-episode link context.

Stable model/run settings live in the resolved YAML config. Link context is
different: it can change per episode and per flow, so the orchestrator writes a
small JSON file and passes only its path to workers.
"""

import json
import os
from bisect import bisect_right


class StepTrace:
    """Step-function trace with O(log n) lookup by absolute time."""

    def __init__(self, initial_value, samples=None):
        self.initial_value = initial_value
        samples = sorted(samples or [], key=lambda item: item[0])
        self.times = [float(t) for t, _value in samples]
        self.values = [value for _t, value in samples]

    def at(self, t):
        idx = bisect_right(self.times, float(t)) - 1
        if idx < 0:
            return self.initial_value
        return self.values[idx]


def write_link_context(path, *, bw_mbps, base_rtt_us, link_schedule,
                       episode=None, slot=None, flow_id=None):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    payload = {
        'version': 1,
        'episode': episode,
        'slot': slot,
        'flow_id': flow_id,
        'base': {
            'bw_mbps': float(bw_mbps),
            'rtt_us': float(base_rtt_us),
        },
        # Keep event keys in the existing schedule language: t seconds from
        # episode start, bw Mbps, delay ms.
        'events': [dict(entry) for entry in (link_schedule or [])],
    }
    tmp = f'{path}.tmp.{os.getpid()}'
    with open(tmp, 'w') as f:
        json.dump(payload, f, separators=(',', ':'))
    os.replace(tmp, path)
    return path


def load_link_context(path=None, *, env_prefix='OC'):
    path = path or os.environ.get(f'{env_prefix}_LINK_CONTEXT_PATH')
    if not path:
        return _env_context(env_prefix)
    try:
        with open(path) as f:
            payload = json.load(f) or {}
    except (OSError, ValueError, TypeError):
        return _env_context(env_prefix)
    return payload if isinstance(payload, dict) else _env_context(env_prefix)


def context_values(path=None, *, env_prefix='OC'):
    ctx = load_link_context(path, env_prefix=env_prefix)
    base = ctx.get('base') if isinstance(ctx.get('base'), dict) else {}
    return (
        float(base.get('bw_mbps', 100.0)),
        float(base.get('rtt_us', 20_000.0)),
        list(ctx.get('events') or []),
    )


def build_step_trace(initial_value, link_schedule, episode_start, field,
                     transform=lambda value: value, warn_prefix='link_context'):
    samples = []
    for entry in (link_schedule or []):
        if 't' not in entry:
            print(f'[{warn_prefix}] WARNING: schedule entry missing "t" key, '
                  f'skipped: {entry}', flush=True)
            continue
        if field not in entry:
            continue
        try:
            t_abs = float(episode_start) + float(entry['t'])
            value = transform(entry[field])
        except (TypeError, ValueError):
            continue
        samples.append((t_abs, value))
    return StepTrace(initial_value, samples)


def _env_context(env_prefix):
    bw_raw = os.environ.get(f'{env_prefix}_LINK_BW', '100')
    rtt_raw = os.environ.get(f'{env_prefix}_BASE_RTT_US', '20000')
    sched_raw = os.environ.get(f'{env_prefix}_LINK_SCHEDULE', '[]')
    try:
        schedule = json.loads(sched_raw)
    except (TypeError, ValueError):
        schedule = []
    try:
        bw_mbps = float(bw_raw)
    except (TypeError, ValueError):
        bw_mbps = 100.0
    try:
        rtt_us = float(rtt_raw)
    except (TypeError, ValueError):
        rtt_us = 20_000.0
    return {
        'version': 1,
        'base': {
            'bw_mbps': bw_mbps,
            'rtt_us': rtt_us,
        },
        'events': list(schedule or []),
    }

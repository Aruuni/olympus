"""Absolute-grid loop pacing for collection workers.

Workers historically paced with ``sleep(interval - elapsed)``, which anchors
each worker's sampling grid to whenever *its own* loop happened to start — a
fixed per-worker phase offset (staggered process startup) between flows
collecting the same episode. Pacing against the shared episode-start clock
instead snaps every worker onto the common grid ``t0 + k*interval``, so
co-active flows sample at the same instants and the learner's
``(group_id, group_step)`` joins are temporally aligned.

Valid because the orchestrator injects one shared ``t0`` (``SAO_EPISODE_START``)
into every worker and ``time.monotonic()`` is system-wide on Linux.
"""

import time


def sleep_to_grid(next_tick, interval_s):
    """Sleep until the next tick on the shared absolute grid; return the
    updated target.

    ``next_tick`` is the previous target — initialise it to the shared
    episode-start ``t0`` before the loop. Each call advances it by one interval;
    if the loop overran the grid point, the missed buckets are skipped rather
    than busy-caught-up (preserving the old "don't sleep when behind" behaviour,
    at the cost of a skipped step, which shows up as a rare missing cell).
    """
    next_tick += interval_s
    now = time.monotonic()
    if next_tick <= now:
        next_tick += ((now - next_tick) // interval_s + 1.0) * interval_s
    else:
        time.sleep(next_tick - now)
    return next_tick

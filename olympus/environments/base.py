"""Backend-agnostic network interface for Olympus environments.

A *backend* turns an episode's link/traffic spec into flows that the
congestion-control workers can attach to. Mininet is the reference backend
(``olympus/environments/mininet/env.py``); alternatives live under
``olympus/environments/<type>/env.py`` and are selected by the environment
``type`` in config and instantiated through ``make_env`` (see
``olympus/environments/__init__.py``).

Flow identity — the invariant every backend must honor:

    Flow ``i`` is addressed by the key ``cport`` (or ``cport + i - 1`` when
    ``unique_cports`` is set). Workers, listeners, and downstream parsers use
    that key to find their flow; what the key binds to is backend-specific.

    *Emulation* backends (real kernel TCP flows — Mininet and friends) bind
    the key literally: it IS the iperf3 client source port. Such a backend
    must:

      * launch real TCP flows whose client-side source port is exactly the
        requested cport (honoring ``unique_cports``);
      * run them where the listener's socket scan can observe them;
      * keep parallel-slot isolation so concurrent instances never collide on
        source ports or emulator resources (see ``instance_id``);
      * write the per-flow iperf artifacts the downstream parsers read
        (``/tmp/iperf_{cport}_{i}.json`` and the matching server file).

    *Simulation* backends (e.g. RayNet) have no kernel sockets: they expose
    their own attach mechanism (RayNet: a flow-service RPC channel) and keep
    the cport purely as the per-flow identity label.

Universal constructor kwargs (a backend may accept extras, but these carry the
shared meaning the orchestrator relies on):

    n               number of sender/receiver flow pairs
    bw              bottleneck bandwidth (Mbps)
    delay           one-way propagation delay (ms)
    qsize           bottleneck queue size (overrides bdp_mult when set)
    bdp_mult        queue size as a multiple of BDP
    loss            bottleneck loss percentage (optional)
    duration        per-episode flow runtime (s)
    cport           per-flow identity key — see the invariant above
    cc_algo         TCP congestion-control name passed to the flow generator
    instance_id     isolation-slot id for parallel runs (None = legacy/no prefix)
    unique_cports   give flow i identity key cport+i-1 instead of a shared cport
    per_flow_delays optional list of per-flow one-way delays (ms)
"""

import threading
import time
from abc import ABC, abstractmethod


class NetworkEnv(ABC):
    """Lifecycle a network backend must implement for the orchestrator.

    Callers drive exactly five methods, in order::

        start()                            bring up backend resources
        setup_environment(link_schedule)   apply the initial scenario and hand
                                           over the mid-episode change schedule
        start_episode(..., episode_start)  launch the flows (non-blocking)
        wait()                             block until the episode finished
        stop()                             tear down

    The scenario is fully declarative from the caller's side: the static link
    (constructor kwargs) and the schedule of mid-episode changes (a list of
    ``{'t': s, 'bw': ..., 'delay': ..., 'loss': ...}`` entries, ``t`` relative
    to the episode anchor) are handed to ``setup_environment``, and *how* they
    are realized is the backend's business. A backend either consumes the
    schedule declaratively (RayNet folds it into the episode config the
    simulator executes internally) or replays it live against the running
    network: for the latter, this base class owns the replay — a background
    thread that applies each entry through ``change_link`` at its scheduled
    offset. A live-replaying backend overrides ``change_link`` and brackets the
    replay with ``_begin_link_schedule`` (end of ``start_episode``) and
    ``_stop_link_schedule`` (top of ``stop``), as MininetEnv does; ``change_link``
    is otherwise optional and defaults to accepting and ignoring the call.

    Timing: ``episode_start`` (a ``time.monotonic`` value) is the anchor the
    caller also exports to workers as ``OC_EPISODE_START``. Reward and state
    plugins locate the current link on the schedule from that anchor, so the
    replay must be anchored there too — anchoring at ``start_episode`` call
    time instead would shift the executed schedule relative to the timeline
    the workers assume.
    """

    # ------------------------------------------------------------------
    # Required backend surface
    # ------------------------------------------------------------------

    @abstractmethod
    def start(self) -> None:
        """Bring up the backend's resources (topology / namespaces / services)."""

    @abstractmethod
    def setup_environment(self, link_schedule=None) -> None:
        """Apply the episode's initial link scenario and take ownership of the
        mid-episode change schedule.

        Extend via ``super().setup_environment(link_schedule)``: the base body
        stores the schedule for the ``_begin_link_schedule`` replay helper.
        Backends that execute the schedule internally consume it here instead.
        """
        self._link_schedule = [dict(e) for e in (link_schedule or [])]

    @abstractmethod
    def start_episode(self, monitor_interval=0.1, start_delays=None,
                      flow_durations=None, episode_start=None) -> None:
        """Launch the flows that realize the episode's traffic, then return.

        Must not block for the episode duration — callers use ``wait`` for
        that. ``start_delays`` optionally staggers flow start times (s, one per
        flow); ``flow_durations`` optionally sets an exact runtime per flow;
        ``episode_start`` is the shared schedule anchor documented above.
        """

    @abstractmethod
    def wait(self) -> None:
        """Block until the episode launched by ``start_episode`` has finished.

        Raise if the episode failed; return normally once it ran to completion.
        """

    @abstractmethod
    def stop(self) -> None:
        """Tear down everything ``start`` created and release all resources."""

    # ------------------------------------------------------------------
    # Optional live-update capability
    # ------------------------------------------------------------------

    def change_link(self, bw=None, delay=None, loss=None) -> None:
        """Live-update bottleneck bandwidth, delay, and/or loss mid-episode.

        Optional capability: the default accepts and ignores the call. A
        backend that can retune the bottleneck in place overrides this; it is
        the primitive the base schedule replay drives, and callers (tests,
        benchmarks) may also invoke it directly. Must be thread-safe — the
        replay calls it from a background thread.
        """

    # ------------------------------------------------------------------
    # Shared link-schedule replay (drives change_link; live backends only)
    # ------------------------------------------------------------------

    def _begin_link_schedule(self, episode_start=None) -> None:
        """Start replaying the schedule registered by ``setup_environment``.

        No-op when no schedule was registered. ``episode_start`` is the shared
        anchor; ``None`` falls back to now, which is only correct for callers
        that do not export an anchor to workers.
        """
        schedule = getattr(self, '_link_schedule', None)
        if not schedule:
            return
        anchor = time.monotonic() if episode_start is None else float(episode_start)
        self._sched_stop = threading.Event()
        self._sched_thread = threading.Thread(
            target=self._replay_link_schedule,
            args=(schedule, anchor, self._sched_stop),
            daemon=True)
        self._sched_thread.start()

    def _stop_link_schedule(self) -> None:
        """Stop the replay thread. Safe to call when it was never started."""
        stop = getattr(self, '_sched_stop', None)
        if stop is not None:
            stop.set()
        thread = getattr(self, '_sched_thread', None)
        if thread is not None:
            thread.join(timeout=2)
            self._sched_thread = None

    def _replay_link_schedule(self, schedule, anchor, stop) -> None:
        for entry in schedule:
            target = anchor + float(entry.get('t', 0.0))
            while not stop.is_set():
                remaining = target - time.monotonic()
                if remaining <= 0:
                    break
                stop.wait(timeout=min(remaining, 0.05))
            if stop.is_set():
                return
            try:
                self.change_link(
                    bw=entry.get('bw'),
                    delay=entry.get('delay'),
                    loss=entry.get('loss'),
                )
                print(f'[env] link change t={time.monotonic() - anchor:.1f}s '
                      f'bw={entry.get("bw")} delay={entry.get("delay")}',
                      flush=True)
            except Exception as e:
                print(f'[env] link change failed: {e}', flush=True)

"""Template Olympus network backend — copy this folder to add a new one.

How to use this template:

  1. Copy ``olympus/environments/_template`` to ``olympus/environments/<type>``
     (e.g. ``netns``, ``containerlab``, ``testbed``). The folder name IS the
     environment ``type`` selected in config / on the orchestrator CLI.
  2. Rename ``TemplateEnv`` to something descriptive and implement the five
     lifecycle methods below (start → setup_environment → start_episode /
     wait → stop). ``change_link`` is optional: override it (and bracket the
     base-class replay helpers as shown) only if your backend can retune the
     bottleneck live; a backend that executes the schedule itself just
     consumes it in ``setup_environment``. Keep ``ENV_CLASS`` pointing at
     your class — ``olympus.environments.make_env`` resolves a backend by
     importing ``olympus.environments.<type>.env`` and instantiating its
     ``ENV_CLASS``.
  3. Honor the flow-identity invariant (see ``olympus/environments/base.py``):
     flow ``i`` is addressed by the cport key. Emulation backends bind it
     literally as the iperf3 client source port so the congestion-control
     listener can attach to each flow; simulation backends keep it as a label
     and provide their own attach mechanism. This is the one invariant you
     cannot skip — the rest of the topology is yours to model however you
     like.

The reference implementation is ``olympus/environments/mininet/env.py``; read
it alongside this file for a concrete, working example of every method.
"""

from olympus.environments.base import NetworkEnv


class TemplateEnv(NetworkEnv):
    """Skeleton backend. Replace every ``raise NotImplementedError`` body.

    The constructor receives the universal kwargs the orchestrator passes to
    every backend (documented in ``olympus/environments/base.py``). Accept and
    store them even if your backend ignores some — callers always pass them.
    Add backend-specific kwargs with defaults so existing configs keep working.
    """

    def __init__(self, n=1, bw=10, delay=20, qsize=None, bdp_mult=1.0,
                 loss=None, duration=60, cport=11111, cc_algo='mutant',
                 instance_id=None, unique_cports=False, per_flow_delays=None,
                 **extra):
        self.n             = n
        self.bw            = bw
        self.delay         = delay
        self.qsize         = qsize
        self.bdp_mult      = bdp_mult
        self.loss          = loss
        self.duration      = duration
        self.cport         = cport          # flow-identity key — see contract
        self.cc_algo       = cc_algo
        self.instance_id   = instance_id     # isolation slot for parallel runs
        self.unique_cports = unique_cports
        self.per_flow_delays = (
            [float(d) for d in per_flow_delays] if per_flow_delays else None)
        # iperf3 server port; offset by slot so parallel instances never clash.
        self.iperf_port = 5201 + (instance_id if instance_id is not None else 0)

    def start(self) -> None:
        """Bring up the topology / namespaces / services.

        Create your sender/receiver endpoints (n pairs) and wire the
        bottleneck. Make this idempotent: clean up any leftovers from a
        previous episode on the same slot first (see MininetEnv.start, which
        deletes stale bridges/interfaces)."""
        raise NotImplementedError('TemplateEnv.start')

    def setup_environment(self, link_schedule=None) -> None:
        """Apply the episode's initial link scenario and take ownership of the
        mid-episode change schedule.

        Configure bw/delay/loss + the BDP-sized queue on the bottleneck (see
        MininetEnv.setup_environment). Call
        ``super().setup_environment(link_schedule)`` first — the base body
        stores the schedule for the live-replay helpers. A backend that
        executes the schedule internally (see RaynetEnv) consumes it here
        instead and never starts the replay."""
        raise NotImplementedError('TemplateEnv.setup_environment')

    def change_link(self, bw=None, delay=None, loss=None) -> None:
        """OPTIONAL — live-update the bottleneck mid-episode (bw/delay/loss).

        Delete this override to inherit the accept-and-ignore default.
        Implement it if your backend can retune in place: update only the
        parameters that are not None, recompute the queue size from the new
        BDP, and stay thread-safe — the base-class schedule replay calls this
        from a background thread. Then start the replay with
        ``self._begin_link_schedule(episode_start)`` at the end of
        ``start_episode`` and stop it with ``self._stop_link_schedule()`` at
        the top of ``stop`` (see MininetEnv)."""
        raise NotImplementedError('TemplateEnv.change_link')

    def start_episode(self, monitor_interval=0.1, start_delays=None,
                      flow_durations=None, episode_start=None) -> None:
        """Launch the flows that realize every episode, then return.

        Must not block for the episode duration — that is wait()'s job.
        ``episode_start`` is the shared schedule anchor (the caller exports
        the same value to workers); pass it to ``_begin_link_schedule`` when
        replaying the schedule live. For an
        emulation backend this is where the cport binding is enforced: for
        each flow ``i`` in 1..n, start an iperf3 client whose source port is::

            client_cport = self.cport + i - 1 if self.unique_cports else self.cport

        and run it with ``--cport=<client_cport> -C <self.cc_algo>`` against the
        matching receiver, writing results to ``/tmp/iperf_{self.cport}_{i}.json``
        (server side: ``/tmp/iperf_server_{self.cport}_{i}.json``) so downstream
        parsers find them. ``start_delays`` staggers per-flow start times (s);
        ``flow_durations`` sets an exact runtime per flow when supplied,
        otherwise derive it from ``self.duration`` minus the start delay.

        See MininetEnv.start_episode for the exact command line to mirror."""
        raise NotImplementedError('TemplateEnv.start_episode')

    def wait(self) -> None:
        """Block until the episode launched by start_episode() has finished.

        Raise if the episode failed. MininetEnv simply sleeps duration + grace;
        a backend with a real completion signal should block on that instead
        (see RaynetEnv.wait)."""
        raise NotImplementedError('TemplateEnv.wait')

    def stop(self) -> None:
        """Tear down everything start() created and release all resources.

        Must be safe to call when start() never ran or already stopped."""
        raise NotImplementedError('TemplateEnv.stop')


# Backend entry point resolved by olympus.environments.make_env.
ENV_CLASS = TemplateEnv

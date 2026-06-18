"""Template Olympus network backend — copy this folder to add a new one.

How to use this template:

  1. Copy ``olympus/environments/_template`` to ``olympus/environments/<type>``
     (e.g. ``netns``, ``containerlab``, ``testbed``). The folder name IS the
     environment ``type`` selected in config / on the orchestrator CLI.
  2. Rename ``TemplateEnv`` to something descriptive and implement the four
     lifecycle methods below. Keep ``ENV_CLASS`` pointing at your class —
     ``olympus.environments.make_env`` resolves a backend by importing
     ``olympus.environments.<type>.env`` and instantiating its ``ENV_CLASS``.
  3. Honor the cport contract (see ``olympus/environments/base.py``): flow
     identity is the iperf3 client source port, so the congestion-control
     listener can attach to each flow. This is the one invariant you cannot
     skip — the rest of the topology is yours to model however you like.

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
        """Bring up the topology / namespaces and apply the initial link.

        Create your sender/receiver endpoints (n pairs), wire the bottleneck,
        and apply bw/delay/loss + the BDP-sized queue. Make this idempotent:
        clean up any leftovers from a previous episode on the same slot first
        (see MininetEnv.start, which deletes stale bridges/interfaces)."""
        raise NotImplementedError('TemplateEnv.start')

    def set_link(self, bw=None, delay=None, loss=None) -> None:
        """Live-update the bottleneck mid-episode (bw/delay/loss).

        Called from a background scheduler thread, so it must be thread-safe.
        Update only the parameters that are not None, recompute the queue size
        from the new BDP, and retune in place. A backend that cannot retune
        live should still accept the call without raising."""
        raise NotImplementedError('TemplateEnv.set_link')

    def run_iperf(self, monitor_interval=0.1, start_delays=None,
                  flow_durations=None) -> None:
        """Launch the iperf3 servers and clients that realize every flow.

        This is where the cport contract is enforced. For each flow ``i`` in
        1..n, start an iperf3 client whose source port is::

            client_cport = self.cport + i - 1 if self.unique_cports else self.cport

        and run it with ``--cport=<client_cport> -C <self.cc_algo>`` against the
        matching receiver, writing results to ``/tmp/iperf_{self.cport}_{i}.json``
        (server side: ``/tmp/iperf_server_{self.cport}_{i}.json``) so downstream
        parsers find them. ``start_delays`` staggers per-flow start times (s);
        ``flow_durations`` sets an exact runtime per flow when supplied,
        otherwise derive it from ``self.duration`` minus the start delay.

        See MininetEnv.run_iperf for the exact command line to mirror."""
        raise NotImplementedError('TemplateEnv.run_iperf')

    def stop(self) -> None:
        """Tear down everything start() created and release all resources.

        Must be safe to call when start() never ran or already stopped."""
        raise NotImplementedError('TemplateEnv.stop')


# Backend entry point resolved by olympus.environments.make_env.
ENV_CLASS = TemplateEnv

"""Off-thread experience pushing for collection workers.

The worker hot loop must run on a fixed control-step cadence (one action per
``interval_ms``). Pushing experience batches to the learner is a *blocking* RPC
on the manager proxy, so doing it inline stalls the loop every ``push_every``
steps and leaves a regularly-spaced gap in the emitted control signal (e.g. the
CWND multiplier). ``AsyncPusher`` moves that RPC onto a dedicated background
thread so the loop only ever does a cheap in-memory enqueue.

Thread-safety: ``multiprocessing`` manager proxies keep a per-thread connection,
so the sender thread calling ``push_exp_batch`` runs on its own connection and
does not race the main thread's ``pull_params``. A single sender thread also
preserves push ordering.
"""

import queue
import threading


class AsyncPusher:
    """Background sender for experience batches.

    ``push_batch(exps)`` / ``push_one(exp)`` are the same callables the worker
    used to invoke inline; here they are called from the sender thread. If a
    batch push raises, we fall back to pushing its experiences one at a time,
    matching the previous inline behaviour.
    """

    def __init__(self, push_batch, push_one, maxsize=0):
        self._push_batch = push_batch
        self._push_one = push_one
        # maxsize=0 is unbounded. Kept large-or-unbounded on purpose: dropping
        # experiences would corrupt training. Backpressure only shows up if the
        # learner endpoint stalls, in which case put() blocks the loop — the
        # same effective throttling the inline version had.
        self._q = queue.Queue(maxsize=maxsize)
        self._stop = object()  # sentinel
        self._thread = threading.Thread(
            target=self._run, name='async-pusher', daemon=True)
        self._thread.start()

    def _run(self):
        while True:
            item = self._q.get()
            if item is self._stop:
                return
            exps = item
            try:
                self._push_batch(exps)
            except Exception:
                for e in exps:
                    try:
                        self._push_one(e)
                    except Exception:
                        pass

    def submit(self, exps):
        """Enqueue a batch (a list of experiences) for background sending."""
        if exps:
            self._q.put(list(exps))

    def close(self, timeout=5.0):
        """Drain outstanding batches and stop the sender thread."""
        self._q.put(self._stop)
        self._thread.join(timeout=timeout)

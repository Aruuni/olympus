"""BDP-stratified replay buffer for the simulation side of mixed collection.

In mixed emulation+simulation training the simulation backend (RayNet/OMNeT++)
finishes low-bandwidth episodes far faster in wall-clock than high-bandwidth
ones, because they carry fewer packets/events. Feeding a single FIFO buffer then
lets the fast low-BDP episodes evict the rare high-BDP experience before the
policy can learn from it, collapsing the learned policy toward low cwnd.

``StratifiedReplay`` fixes that by partitioning the simulation buffer into one
equal-capacity sub-buffer per **bandwidth-delay-product (BDP) class**. Pushes are
routed by the episode's BDP; each sampled batch is split roughly evenly across
the non-empty classes. Because each class keeps its own reserved slice, fast
low-BDP episodes can no longer evict high-BDP experience, and every regime is
represented in every minibatch regardless of how quickly it is produced.

The BDP key is a scalar proportional to the true bandwidth-delay product
(``bw_mbps * rtt_ms``); the learner bins it with configurable edges. Only the
simulation side is stratified — emulation stays a single buffer (see
:class:`olympus.common.mixed_replay.MixedReplay`).
"""

import bisect

from olympus.common.mixed_replay import _merge


class StratifiedReplay:
    """One replay buffer per BDP class, sampled evenly across non-empty classes.

    Parameters
    ----------
    factory : callable() -> buffer
        Builds one replay buffer. Called ``len(edges) + 1`` times (one per BDP
        class), so every class gets a buffer of whatever capacity the factory
        closes over. To keep the simulation side's *total* capacity comparable to
        an unstratified buffer, give the factory ``capacity // (len(edges) + 1)``.
    edges : sequence of float
        Sorted BDP bin boundaries. ``k`` edges create ``k + 1`` classes; a value
        ``v`` lands in class ``bisect_right(edges, v)``.
    """

    def __init__(self, factory, edges):
        self.edges = sorted(float(e) for e in edges)
        self.n_bins = len(self.edges) + 1
        self.bins = [factory() for _ in range(self.n_bins)]

    # ---- routing + sizes -------------------------------------------------
    def bin_index(self, value):
        """Class index for a BDP ``value`` (``None`` -> class 0)."""
        if value is None:
            return 0
        return bisect.bisect_right(self.edges, float(value))

    def push(self, exp, value=None):
        self.bins[self.bin_index(value)].push(exp)

    def size(self):
        return sum(b.size() for b in self.bins)

    def bin_sizes(self):
        return [b.size() for b in self.bins]

    def n_trajs(self):
        return sum(b.n_trajs() for b in self.bins if hasattr(b, 'n_trajs'))

    # ---- batch composition ----------------------------------------------
    def split(self, total):
        """Split a batch ``total`` evenly across the non-empty classes.

        Each non-empty class gets ``total // n_active`` samples; the remainder is
        handed to the largest classes (which can always supply it). Returns a
        per-class allocation list of length :attr:`n_bins`.
        """
        total = int(total)
        alloc = [0] * self.n_bins
        active = [i for i in range(self.n_bins) if self.bins[i].size() > 0]
        if not active or total <= 0:
            return alloc
        base, rem = divmod(total, len(active))
        for i in active:
            alloc[i] = base
        for i in sorted(active, key=lambda i: self.bins[i].size(), reverse=True)[:rem]:
            alloc[i] += 1
        return alloc

    # ---- sample / snapshot forwarding -----------------------------------
    def __getattr__(self, name):
        # Only for undefined attributes (e.g. sample, sample_chunks,
        # sample_transitions, snapshot_rewards); mirrors MixedReplay.
        if not (name.startswith('sample') or name.startswith('snapshot')):
            raise AttributeError(name)

        def _forward(total, *args, **kwargs):
            alloc = self.split(total)
            result = None
            for b, n in zip(self.bins, alloc):
                if n > 0:
                    result = _merge(result, getattr(b, name)(n, *args, **kwargs))
            return result

        return _forward

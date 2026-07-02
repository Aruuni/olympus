"""Mixed emulation + simulation replay buffer.

Used by single-agent off-policy learners (orca, td3, mbpo_td3, dreamer_v3) when
``mixed_collection`` is enabled. It wraps **two instances of the learner's own
replay buffer** — one fed by the emulation backend, one by the simulation
backend — both built with the *same* configurable capacity. Pushes are routed by
the experience's source tag; samples are drawn from both buffers and merged so
each training minibatch is a configurable blend of the two.

The wrapper mirrors the underlying buffer's interface: ``push`` gains a ``source``
argument, ``size``/``n_trajs`` report the combined totals, and any ``sample*`` /
``snapshot*`` call is split by the mix fraction across the two buffers and merged
back together. Because each buffer is the learner's native ``ReplayBuffer``, the
algorithm's transition/sequence format is preserved unchanged.

Warm-up policy: training may begin as soon as *either* buffer reaches
``min_replay`` (see :meth:`MixedReplay.ready`), and samples are only drawn from
buffers that currently hold data, so the slower backend never stalls learning.
"""

import numpy as np

_EMU = 'emulation'
_SIM = 'simulation'


def mixed_settings(cfg):
    """Parse the ``mixed_collection`` block from a resolved config.

    Returns ``(enabled, emulation_fraction, buffer_capacity_override)``.
    ``buffer_capacity_override`` is ``None`` when the learner should fall back to
    its own ``replay_capacity`` (both buffers always get the same capacity).
    """
    mc = (cfg or {}).get('mixed_collection') or {}
    enabled = bool(mc.get('enabled', False))
    mix = mc.get('mix') or {}
    frac = float(mix.get('emulation_batch_fraction', 0.5))
    cap = mix.get('buffer_capacity')
    return enabled, frac, (int(cap) if cap is not None else None)


def mixed_buffer_capacities(cfg, default_capacity):
    """Total replay capacity for each collection side, ``(emu_cap, sim_cap)``.

    Resolution order per side: ``mixed_collection.<side>.buffer_capacity`` ->
    shared ``mixed_collection.mix.buffer_capacity`` -> ``default_capacity`` (the
    learner's ``replay_capacity``). ``sim_cap`` is the simulation side's *total*
    footprint; when stratified the learner divides it across BDP classes.
    """
    mc = (cfg or {}).get('mixed_collection') or {}
    shared = (mc.get('mix') or {}).get('buffer_capacity')
    base = int(shared) if shared is not None else int(default_capacity)

    def _side(name):
        v = (mc.get(name) or {}).get('buffer_capacity')
        return int(v) if v is not None else base

    return _side('emulation'), _side('simulation')


def sim_strata_edges(cfg):
    """BDP bin edges for the simulation buffer, or ``None`` if not configured.

    Read from ``mixed_collection.simulation.strata.bdp_edges``. When present the
    learner should build the simulation side as a
    :class:`olympus.common.stratified_replay.StratifiedReplay` so fast low-BDP
    episodes cannot evict high-BDP experience. Returns a sorted list of floats.
    """
    mc = (cfg or {}).get('mixed_collection') or {}
    sim = mc.get('simulation') or {}
    strata = sim.get('strata') or {}
    edges = strata.get('bdp_edges')
    if not edges:
        return None
    return sorted(float(e) for e in edges)


def split_source(source):
    """Parse a worker experience tag into ``(base, bdp)``.

    Workers tag simulation experiences ``'simulation'`` or ``'simulation|<bdp>'``
    (the trailing scalar is proportional to bandwidth-delay product). Emulation
    is ``'emulation'``. Unknown / ``None`` tags route to the simulation side.
    """
    if source == _EMU:
        return _EMU, None
    if isinstance(source, str) and source.startswith(_SIM):
        rest = source[len(_SIM):]
        if rest.startswith('|'):
            try:
                return _SIM, float(rest[1:])
            except ValueError:
                return _SIM, None
    return _SIM, None


def _concat(x, y):
    if isinstance(x, np.ndarray):
        return np.concatenate([x, y], axis=0)
    return list(x) + list(y)


def _merge(a, b):
    """Merge two sample results (dict / tuple / list / ndarray / None)."""
    if a is None:
        return b
    if b is None:
        return a
    if isinstance(a, dict):
        return {k: _concat(a[k], b[k]) for k in a}
    if isinstance(a, tuple):
        return tuple(_concat(x, y) for x, y in zip(a, b))
    if isinstance(a, list):
        return [_concat(x, y) for x, y in zip(a, b)]
    return _concat(a, b)


class MixedReplay:
    """Two same-capacity replay buffers blended by a fixed emulation fraction.

    Parameters
    ----------
    factory : callable() -> buffer
        Builds one replay buffer. Called for the emulation buffer and (when
        ``sim_strata`` is ``None``) the simulation buffer, so both share whatever
        capacity the factory closes over.
    emulation_fraction : float
        Target share of each sampled batch drawn from the emulation buffer,
        clamped to ``[0, 1]``. The remainder comes from the simulation buffer.
    sim_strata : sequence of float, optional
        BDP bin edges. When given, the simulation side becomes a
        :class:`StratifiedReplay` with one sub-buffer per BDP class so fast
        low-BDP episodes cannot evict high-BDP experience. Emulation is never
        stratified.
    sim_factory : callable() -> buffer, optional
        Factory for the simulation buffer(s); defaults to ``factory``. Pass a
        smaller-capacity factory when stratifying so the per-class buffers sum to
        roughly the emulation buffer's capacity.
    """

    def __init__(self, factory, emulation_fraction, sim_strata=None,
                 sim_factory=None):
        self.emu = factory()
        sim_factory = sim_factory or factory
        if sim_strata:
            from olympus.common.stratified_replay import StratifiedReplay
            self.sim = StratifiedReplay(sim_factory, sim_strata)
        else:
            self.sim = sim_factory()
        self._sim_stratified = bool(sim_strata)
        self.emulation_fraction = float(min(1.0, max(0.0, emulation_fraction)))

    # ---- routing + sizes -------------------------------------------------
    def push(self, exp, source=None):
        base, bdp = split_source(source)
        if base == _EMU:
            self.emu.push(exp)
        elif self._sim_stratified:  # route into the matching BDP class
            self.sim.push(exp, bdp)
        else:
            self.sim.push(exp)

    def size(self):
        return self.emu.size() + self.sim.size()

    def size_emu(self):
        return self.emu.size()

    def size_sim(self):
        return self.sim.size()

    def n_trajs(self):
        e = self.emu.n_trajs() if hasattr(self.emu, 'n_trajs') else 0
        s = self.sim.n_trajs() if hasattr(self.sim, 'n_trajs') else 0
        return e + s

    def ready(self, min_replay):
        """True once *either* buffer has reached ``min_replay`` transitions."""
        return self.emu.size() >= min_replay or self.sim.size() >= min_replay

    # ---- batch composition ----------------------------------------------
    def split(self, total):
        """Split a batch ``total`` into ``(n_emu, n_sim)``.

        Honours :attr:`emulation_fraction` when both buffers hold data; otherwise
        routes the whole batch to whichever buffer currently has data.
        """
        total = int(total)
        has_emu = self.emu.size() > 0
        has_sim = self.sim.size() > 0
        if has_emu and has_sim:
            n_emu = int(round(self.emulation_fraction * total))
            n_emu = max(0, min(total, n_emu))
            return n_emu, total - n_emu
        if has_emu:
            return total, 0
        if has_sim:
            return 0, total
        return 0, 0

    # ---- sample / snapshot forwarding -----------------------------------
    def __getattr__(self, name):
        # Only triggered for attributes not defined above (e.g. sample,
        # sample_chunks, sample_transitions, sample_states, snapshot_rewards).
        if not (name.startswith('sample') or name.startswith('snapshot')):
            raise AttributeError(name)

        def _forward(total, *args, **kwargs):
            n_emu, n_sim = self.split(total)
            re = getattr(self.emu, name)(n_emu, *args, **kwargs) if n_emu > 0 else None
            rs = getattr(self.sim, name)(n_sim, *args, **kwargs) if n_sim > 0 else None
            return _merge(re, rs)

        return _forward

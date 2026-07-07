"""Multi-environment experience-collection replay buffer.

Used by single-agent off-policy learners (orca, td3, mbpo_td3, dreamer_v3) when
``experience_collection`` is enabled. The config block declares one entry per
collection environment type; each entry owns a replay buffer (an instance of
the learner's native buffer class), a share of every training minibatch, and
optionally a BDP-stratified layout:

    experience_collection:
      enabled: true
      mininet:
        environment_setup: dynamic
        n_parallel: 20
        buffer_capacity: 400000
        batch_fraction: 0.5
      raynet:
        environment_setup: orca_static
        n_parallel: 10
        buffer_capacity: 400000
        batch_fraction: 0.5
        strata:
          n_buckets: 20

Any number of environment blocks may be declared. Every block must declare a
``batch_fraction`` and the fractions must sum to 1, otherwise the run aborts
(see :func:`validate_batch_fractions`).

Pushes are routed by the experience's source tag — the environment type,
optionally suffixed ``'|<bdp>'`` by simulation workers so a stratified group
can route the transition into its BDP class. Samples are drawn from every
buffer and merged so each training minibatch is the configured blend. Because
each buffer is the learner's native ``ReplayBuffer``, the algorithm's
transition/sequence format is preserved unchanged.

Warm-up policy: training may begin as soon as *any* group's buffer reaches
``min_replay`` (see :meth:`MixedReplay.ready`), and samples are only drawn from
buffers that currently hold data, so a slower backend never stalls learning.
"""

import numpy as np

_RESERVED_KEYS = frozenset({'enabled'})
# Source tags emitted before groups were keyed by environment type.
_LEGACY_ALIASES = {'emulation': 'mininet', 'simulation': 'raynet'}


def collection_groups(cfg):
    """Parse the ``experience_collection`` block into ``(enabled, groups)``.

    ``groups`` maps environment type -> its config block, in declaration order
    (every mapping key except ``enabled``). Empty when the block is absent.
    """
    ec = (cfg or {}).get('experience_collection') or {}
    enabled = bool(ec.get('enabled', False))
    groups = {k: (v or {}) for k, v in ec.items()
              if k not in _RESERVED_KEYS and isinstance(v, (dict, type(None)))}
    return enabled, groups


def group_strata_edges(block):
    """BDP bin edges for one collection group's block, or ``None``.

    Read from ``<group>.strata.bdp_edges``. When present the group's buffer is
    built as a :class:`olympus.common.stratified_replay.StratifiedReplay` so
    fast low-BDP episodes cannot evict high-BDP experience. Returns a sorted
    list of floats.
    """
    strata = (block or {}).get('strata') or {}
    edges = strata.get('bdp_edges')
    if not edges:
        return None
    return sorted(float(e) for e in edges)


def validate_batch_fractions(groups):
    """Fail hard unless the groups' ``batch_fraction`` values sum to 1.

    Every declared collection environment must carry an explicit
    ``batch_fraction``; together they must sum to 1 (within float tolerance).
    Raises :class:`ValueError` otherwise so a misconfigured run stops instead
    of silently training on an unintended data blend.
    """
    missing = [name for name, block in groups.items()
               if (block or {}).get('batch_fraction') is None]
    if missing:
        raise ValueError(
            'experience_collection: missing batch_fraction for '
            f'{missing}; every environment must declare one and together '
            'they must sum to 1')
    total = sum(float(block['batch_fraction']) for block in groups.values())
    if abs(total - 1.0) > 1e-6:
        detail = ' '.join(f'{name}={float(block["batch_fraction"])}'
                          for name, block in groups.items())
        raise ValueError(
            f'experience_collection: batch_fraction values sum to {total} '
            f'({detail}); they must sum to 1')


def split_source(source):
    """Parse a worker experience tag into ``(group, bdp)``.

    Workers tag experiences with their collection environment type (e.g.
    ``'mininet'``, ``'raynet'``); simulation workers append ``'|<bdp>'`` (a
    scalar proportional to bandwidth-delay product) so a stratified group can
    route the transition into its BDP class. Returns ``(None, None)`` for
    missing tags.
    """
    if not isinstance(source, str) or not source:
        return None, None
    base, sep, rest = source.partition('|')
    bdp = None
    if sep:
        try:
            bdp = float(rest)
        except ValueError:
            bdp = None
    return base, bdp


def build_mixed_replay(cfg, buffer_factory, default_capacity,
                       log_prefix='[learner]'):
    """Build a :class:`MixedReplay` from ``experience_collection``, or ``None``.

    Returns ``None`` when the block is disabled or absent (the learner should
    fall back to its plain replay buffer).

    ``buffer_factory`` is ``callable(capacity) -> buffer`` building one native
    replay buffer. Each group's ``buffer_capacity`` is its *total* footprint
    (falling back to ``default_capacity``, the learner's ``replay_capacity``);
    a stratified group splits it evenly across its BDP classes.
    """
    enabled, groups = collection_groups(cfg)
    if not enabled or not groups:
        return None
    validate_batch_fractions(groups)

    specs = {}
    for name, block in groups.items():
        cap_total = int(block.get('buffer_capacity') or default_capacity)
        edges = group_strata_edges(block)
        n_bins = (len(edges) + 1) if edges else 1
        cap = max(1, cap_total // n_bins)
        specs[name] = {
            'factory': (lambda c=cap: buffer_factory(c)),
            'fraction': block.get('batch_fraction'),
            'strata': edges,
        }
    replay = MixedReplay(specs)

    desc = []
    for name, block in groups.items():
        cap_total = int(block.get('buffer_capacity') or default_capacity)
        edges = group_strata_edges(block)
        s = f'{name}: fraction={replay.fractions[name]:.3f} cap={cap_total}'
        if edges:
            n_bins = len(edges) + 1
            s += f' ({n_bins} BDP bins x {max(1, cap_total // n_bins)})'
        desc.append(s)
    print(f'{log_prefix} experience collection enabled | ' + ' | '.join(desc),
          flush=True)
    return replay


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
    """One replay buffer per collection environment, blended by batch fraction.

    Parameters
    ----------
    groups : dict
        Ordered mapping of group name (the environment type) to a spec dict:

        - ``factory``: ``callable() -> buffer`` building the group's native
          replay buffer (or, when stratified, one per-class sub-buffer).
        - ``fraction``: target share of each sampled batch. Normalized across
          all groups; ``None`` / missing counts as 1.0 (an equal share).
        - ``strata``: optional BDP bin edges. When given the group's buffer is
          a :class:`StratifiedReplay` with one sub-buffer per BDP class so fast
          low-BDP episodes cannot evict high-BDP experience.
    """

    def __init__(self, groups):
        if not groups:
            raise ValueError('MixedReplay needs at least one collection group')
        self.bufs = {}
        self._stratified = set()
        fractions = {}
        for name, spec in groups.items():
            edges = spec.get('strata')
            if edges:
                from olympus.common.stratified_replay import StratifiedReplay
                self.bufs[name] = StratifiedReplay(spec['factory'], edges)
                self._stratified.add(name)
            else:
                self.bufs[name] = spec['factory']()
            frac = spec.get('fraction')
            fractions[name] = max(0.0, 1.0 if frac is None else float(frac))
        total = sum(fractions.values())
        if total <= 0.0:
            fractions = {n: 1.0 for n in fractions}
            total = float(len(fractions))
        self.fractions = {n: f / total for n, f in fractions.items()}

    # ---- routing + sizes -------------------------------------------------
    def _resolve(self, base):
        """Map a source tag onto a declared group.

        Exact match first, then the pre-rename aliases ('emulation' ->
        'mininet', 'simulation' -> 'raynet'); anything else falls back to the
        first declared group so no experience is ever dropped.
        """
        if base in self.bufs:
            return base
        alias = _LEGACY_ALIASES.get(base)
        if alias in self.bufs:
            return alias
        return next(iter(self.bufs))

    def push(self, exp, source=None):
        base, bdp = split_source(source)
        name = self._resolve(base)
        if name in self._stratified:  # route into the matching BDP class
            self.bufs[name].push(exp, bdp)
        else:
            self.bufs[name].push(exp)

    def size(self):
        return sum(b.size() for b in self.bufs.values())

    def sizes(self):
        """Per-group transition counts, ``{group: size}`` in declaration order."""
        return {n: b.size() for n, b in self.bufs.items()}

    def n_trajs(self):
        return sum(b.n_trajs() for b in self.bufs.values()
                   if hasattr(b, 'n_trajs'))

    def ready(self, min_replay):
        """True once *any* group's buffer has reached ``min_replay`` transitions."""
        return any(b.size() >= min_replay for b in self.bufs.values())

    # ---- batch composition ----------------------------------------------
    def split(self, total):
        """Split a batch ``total`` into ``{group: count}``.

        Honours :attr:`fractions`, renormalized over the groups that currently
        hold data, with largest-remainder rounding so the counts sum to
        ``total`` exactly.
        """
        total = int(total)
        counts = {n: 0 for n in self.bufs}
        live = [n for n, b in self.bufs.items() if b.size() > 0]
        if not live or total <= 0:
            return counts
        wsum = sum(self.fractions[n] for n in live)
        if wsum <= 0.0:
            weights = {n: 1.0 / len(live) for n in live}
        else:
            weights = {n: self.fractions[n] / wsum for n in live}
        raw = {n: weights[n] * total for n in live}
        for n in live:
            counts[n] = int(raw[n])
        rem = total - sum(counts.values())
        for n in sorted(live, key=lambda k: raw[k] - int(raw[k]), reverse=True):
            if rem <= 0:
                break
            counts[n] += 1
            rem -= 1
        return counts

    # ---- sample / snapshot forwarding -----------------------------------
    def __getattr__(self, name):
        # Only triggered for attributes not defined above (e.g. sample,
        # sample_chunks, sample_transitions, sample_states, snapshot_rewards).
        if not (name.startswith('sample') or name.startswith('snapshot')):
            raise AttributeError(name)

        def _forward(total, *args, **kwargs):
            out = None
            for gname, n in self.split(total).items():
                if n > 0:
                    out = _merge(out, getattr(self.bufs[gname], name)(
                        n, *args, **kwargs))
            return out

        return _forward

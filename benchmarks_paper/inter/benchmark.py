#!/usr/bin/env python3
"""Paper-faithful inter-RTT fairness benchmark (IFIP 2026, Figure 4).

Flow 1 is fixed at ``fixed_flow_rtt_ms`` and flow 2 sweeps ``rtt_ms_values``;
the queue is sized to 0.2x/1x/4x the joining flow's BDP.  All heavy lifting
lives in :mod:`benchmarks_paper.common`; this driver only pins the suite and
the figure renderer.
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks_paper.common import run


def _plot(config_path: str) -> int:
    from benchmarks_paper.inter.plot import plot_from_config
    return plot_from_config(config_path)


def main(argv=None) -> int:
    config = str(Path(__file__).with_name('config.yaml'))
    return run(config, 'inter_rtt', _plot, argv)


if __name__ == '__main__':
    raise SystemExit(main())

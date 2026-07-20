#!/usr/bin/env python3
"""Paper-faithful intra-RTT fairness benchmark (IFIP 2026, Figure 5).

Both flows share the swept RTT and the queue is 1x BDP.  Alongside the goodput
ratio the runner records a per-cell delay ratio (SRTT / reference RTT) for the
paper's delay figure.
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks_paper.common import run


def _plot(config_path: str) -> int:
    from benchmarks_paper.intra.plot import plot_from_config
    return plot_from_config(config_path)


def main(argv=None) -> int:
    config = str(Path(__file__).with_name('config.yaml'))
    return run(config, 'intra_rtt', _plot, argv)


if __name__ == '__main__':
    raise SystemExit(main())

#!/usr/bin/env python3
"""Paper-faithful efficiency benchmark (mininettestbed figure-9 setup).

Four same-protocol flows join a 100 Mbps dumbbell staggered 25 s apart, each
running 125 s, at 20 ms and 200 ms RTT and 0.2x/1x/4x BDP queues.  Each run
records normalized throughput (aggregate goodput / link BW), normalized delay
(pooled SRTT / RTT over each flow's 100 s window), and the retransmission
rate.  The figure is the paper's efficiency scatter: normalized throughput vs
normalized delay with per-run confidence ellipses, one figure per queue size.
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks_paper.common import run


def _plot(config_path: str) -> int:
    from benchmarks_paper.efficiency.plot import plot_from_config
    return plot_from_config(config_path)


def main(argv=None) -> int:
    config = str(Path(__file__).with_name('config.yaml'))
    return run(config, 'efficiency', _plot, argv)


if __name__ == '__main__':
    raise SystemExit(main())

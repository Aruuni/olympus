# IFIP 2026 RTT fairness benchmarks

This directory reproduces Figures 4 and 5 from [*An Experimental Study of
Congestion Control in LEO Satellite Networks*](https://dl.ifip.org/db/conf/networking/networking2026/1571261512.pdf)
(Mihai Mazilu et al.), each as its own self-contained benchmark:

```
benchmarks_paper/
  config.yaml      # MASTER: the shared approach list
  paper_style.py   # shared SciencePlots/LaTeX styling (matches core/plotting.py)
  runtime.py       # self-contained trial engine (no orchestrator)
  common.py        # suite driver: case grid, resume, summaries
  plot_run.py      # per-run diagnostic: all flows on one time axis
  inter/           # Figure 4 — inter-RTT fairness
    benchmark.py  config.yaml  plot.py  data/
  intra/           # Figure 5 — intra-RTT fairness (goodput + delay ratio)
    benchmark.py  config.yaml  plot.py  data/
```

Approaches are defined **once** in the master `config.yaml`; each suite's
`config.yaml` points at it with `approaches_config: ../config.yaml` and owns
every dumbbell parameter itself. Each suite owns its `data/` tree and
`plot.py`, so the two experiments run and plot independently.

The benchmarks are **self-contained**: `runtime.py` drives every trial
directly (MininetEnv + iperf3 / Orca scripts / per-flow RL listeners) with no
`olympus.orchestrator` and no imports from the `benchmarks/` siblings — only
the shared `olympus.common` helpers and the emulator.

**Both flows always run the same protocol**, as in the paper. For Olympus
model approaches every flow gets its own listener + worker loading the same
checkpoint deterministically (`unique_cports` gives each flow its own client
port, so each listener attaches to exactly one flow). A trial fails loudly if
an RL worker did not attach to every flow, so a flow can never silently fall
back to the kernel CC.

## Reproduced setup

Both suites use a 100 Mbps dumbbell with a TBF bottleneck and drop-tail FIFO
queue, two flows, five repetitions, one-second receiver goodput samples, and
the paper's minimum/maximum goodput ratio. Flow 1 starts at 0 s, flow 2 joins
at 50 s, both stop at 200 s, and scoring pools the final 100 s. At run start the
driver temporarily raises `tcp_rmem`/`tcp_wmem` with the authors' 22x headroom
multiplier and disables NIC offloads (restored on exit).

- **inter** fixes flow 1 at 20 ms, sweeps flow 2 over 20–200 ms, and sizes the
  queue to 0.2x, 1x, and 4x the joining flow's BDP.
- **intra** sweeps both flows together at a 1x BDP queue. It also records a
  per-cell delay ratio (SRTT / reference RTT) for the paper's delay figure.

## Run

Dry-run to validate the matrix and checkpoints (no sudo):

```bash
./venv_training/bin/python -m benchmarks_paper.inter.benchmark --dry-run
./venv_training/bin/python -m benchmarks_paper.intra.benchmark --dry-run
```

Run a suite (optionally selecting approaches):

```bash
sudo -E env PATH="$PATH" HOME="$HOME" \
  ./venv_training/bin/python -m benchmarks_paper.inter.benchmark

sudo -E env PATH="$PATH" HOME="$HOME" \
  ./venv_training/bin/python -m benchmarks_paper.intra.benchmark --approach orca-olympus
```

Completed cells are resumed rather than overwritten. Aggregates land in
`<suite>/data/summary.csv`; figures are written as PDF and PNG in
`<suite>/data/figures/`, in the paper's SciencePlots/LaTeX style:

- inter: `goodput_inter_rtt_qmult{0.2,1,4}.{pdf,png}`
- intra: `goodput_ratio_intra_rtt_1.{pdf,png}` and `delay_intra_rtt_qmult1.{pdf,png}`

Regenerate only the figures from existing data:

```bash
./venv_training/bin/python -m benchmarks_paper.inter.plot
./venv_training/bin/python -m benchmarks_paper.intra.plot
```

## Styling

Figures use SciencePlots (`pip install scienceplots`) with LaTeX text, matching
the paper's `core/plotting.py` palette. Each approach in `config.yaml` may set
`protocol: <name>` to adopt the exact paper colour/marker (cubic, orca, bbr3,
sage, astraea, …), or override `color`/`marker` directly; the legend uses
`plot_label`.

## External runtimes

The Astraea paper baseline requires the patched `astraea` kernel CC plus the
local `astraea_listener`. The original Orca baseline requires a built Orca
checkout containing `sender.sh`, `receiver.sh`, and
`rl-module/{orca-server-mahimahi,clientThr}`. Set `ORCA_INSTALL_DIR` or add an
`orca.install_dir` override in the suite's `config.yaml` if it is not installed
at `~/mininettestbed/CC/Orca`.

When the single-agent DREAMER model is trained, fill its `CHANGE_ME` values in
each `config.yaml` and set `enabled: true`; it then joins the matrix and plots
without code changes.

# Multi-Agent Benchmarks

This top-level folder runs the same benchmark suites against learned models
and kernel congestion controls configured in `benchmarks/config.yaml`.
Learned approaches own a resolved training `config` and `checkpoint`; kernel
approaches only name their `kernel_cc`. Suite configs contain experiment
settings shared by both.

## Suites

- `benchmark_responsiveness`: one flow under changing BW/RTT.
- `benchmark_responsive_fairness`: 2/4/5/7 flows join evenly over 45 seconds,
  then share a 15-second-cadence changing link for 300 seconds.
- `benchmark_fairness`: two flows, with flow 2 joining at 20 seconds.
- `benchmark_inter_rtt_fairness`: the same join test with unequal RTTs.
- `benchmark_convergence_4flow`: four flows joining at 0, 25, 50, and 75 seconds.

## Validate

```bash
./venv_training/bin/python benchmarks/benchmark_responsiveness/responsiveness.py \
  --config benchmarks/benchmark_responsiveness/config.yaml --dry-run
./venv_training/bin/python \
  benchmarks/benchmark_responsive_fairness/responsive_fairness.py --dry-run
./venv_training/bin/python benchmarks/benchmark_fairness/fairness.py --dry-run
./venv_training/bin/python benchmarks/benchmark_inter_rtt_fairness/inter_rtt_fairness.py --dry-run
./venv_training/bin/python benchmarks/benchmark_convergence_4flow/convergence_4flow.py --dry-run
```

## Run

Run benchmarks only when no Olympus training job is active because Mininet,
listeners, iperf, and cleanup are host-global resources.

Run every configured approach through all five suites:

```bash
sudo -E env PATH="$PATH" HOME="$HOME" ./venv_training/bin/python benchmarks/run_all.py
```

By default, every approach in `benchmarks/config.yaml` is checked. Select a subset
with one or more `--approach <data_folder-or-name>` arguments. The script runs
suites sequentially and existing complete results are skipped by the individual
benchmark runners. Validate the full sequence without Mininet using:

```bash
./venv_training/bin/python benchmarks/run_all.py --dry-run
```

Run individual suites directly when needed:

```bash
sudo -E env PATH="$PATH" HOME="$HOME" \
  ./venv_training/bin/python benchmarks/benchmark_responsiveness/responsiveness.py \
    --config benchmarks/benchmark_responsiveness/config.yaml

sudo -E env PATH="$PATH" HOME="$HOME" \
  ./venv_training/bin/python \
    benchmarks/benchmark_responsive_fairness/responsive_fairness.py
```

Use the analogous command for the other runner scripts. Results are
written beneath each suite's `data/` directory.

## Kernel congestion controls

Kernel baselines use this compact approach form:

```yaml
- data_folder: cubic
  kind: kernel
  kernel_cc: cubic
  plot_label: CUBIC
```

The default config includes CUBIC, Reno, BBR, BBRv1, Westwood+, Veno, Vegas,
YeAH, CDG, BIC, H-TCP, Hybla, HighSpeed TCP, and Illinois. Load the modular
algorithms manually with:

```bash
bash kernel/ins_all.sh
```

The runners also attempt `modprobe tcp_<kernel_cc>` once before execution and
report the kernel's available CC list if loading fails.

Kernel runs do not start an Olympus listener and do not create RL state logs.
Their per-run plots use saved iperf3 data only: receiver goodput, client TCP
RTT where iperf3 exposes it, and aggregate/fairness values derived from those
samples. Raw files are retained as `iperf_client_flowN.json` and
`iperf_receiver_flowN.json` (the responsiveness suite uses
`iperf_flowN.json` for its client capture).

Run only selected kernel baselines with:

```bash
sudo -E env PATH="$PATH" HOME="$HOME" \
  ./venv_training/bin/python benchmarks/run_all.py \
    --approach cubic --approach bbr
```

Responsive-fairness raw runs remain under
`benchmark_responsive_fairness/data/<model>/`. Experiment execution does not
create plots. Generate the figures separately with:

```bash
./venv_training/bin/python \
  benchmarks/benchmark_responsive_fairness/plot.py
```

`plot.py` is the single plot script and `figures.yaml` its single config: it
renders exactly the figures declared there and nothing else. Each figure entry
names a list of `data_folders` and produces one 3-panel PDF under
`benchmark_responsive_fairness/data/figures/<name>.pdf`: aggregate Goodput CDF |
SRTT CDF | Jain fairness CDF (7 flows), pooled across every flow count, with
Scheduled BW / Base RTT references. Plot labels are pulled from the shared
`benchmarks/config.yaml` approaches, so a folder is drawn with the same label
used everywhere else. Add or remove a `data_folder` in `figures.yaml` to change
what a figure shows.

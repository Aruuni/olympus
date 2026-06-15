# Quickstart: Training and Benchmarks

Both training and benchmarks drive Mininet, so they need root and the project
virtualenv. Always launch with `sudo -E` and `./venv_training/bin/python`, and
never run training and benchmarks at the same time — Mininet, the listeners,
iperf, and cleanup are host-global resources.

If a previous run died uncleanly, reset Mininet first:

```bash
sudo mn -c
```

## Training (`olympus/train.py`)

`train.py` wraps `olympus/orchestrator.py` with a condensed live interface:
an episode progress bar with ETA, recent returns, learner telemetry, and
checkpoint age. The full raw orchestrator output is preserved in
`olympus/logs/train_<timestamp>.log`.

Start a training session:

```bash
sudo -E ./venv_training/bin/python olympus/train.py --config olympus/config.yaml
```

What gets trained is controlled entirely by `olympus/config.yaml`:

- `runtime.algorithm` picks the mode — multi-agent algorithms (`ma_dreamer`,
  `mat`) train over the `sweep` flow counts; single-agent algorithms
  (`dreamer_v3`, `td3`, `orca`, …) run the selected `environment`, and any
  environment with more than one flow becomes lagged self-play automatically.
- `runtime.reward`, `runtime.state`, and `runtime.action` select the reward,
  state, and action plugins.
- `environment.name` selects the environment file from
  `olympus/environments/` (`multiflow_interleave`, `dynamic`, `static`,
  `lagged_fairness`).

### Selecting An Action

Set `runtime.action` to `cwnd_multiplier` (the existing `0.5x` to `2.0x`
default) or `astraea` (small relative changes controlled by
`actions.astraea.step`, default `0.025`).

Action settings live in the top-level `actions:` block. CWND safety limits
remain in each algorithm's `agent:` block. Checkpoints record their action
mapping and cannot be resumed with a different selection.

Useful flags:

| Flag | Effect |
| --- | --- |
| `--episodes N` | Override the episode count from the config |
| `--environment <name-or-yaml>` | Override the environment |
| `--python <path>` | Interpreter for the orchestrator (defaults to `paths.py` from the config) |
| `-v` / `--verbose` | Stream the raw orchestrator output instead of the dashboard (still logged) |

Press Ctrl-C **once** to shut down gracefully — the orchestrator saves a final
checkpoint while train.py keeps draining its output. A second Ctrl-C kills it
hard.

Checkpoints and resolved configs land under `olympus/models/<run_dir>/`; the
run directory is printed as an event line (`run dir ...`) when training starts.

## Benchmarks (`benchmarks/run_all.py`)

`run_all.py` runs every learned or kernel approach configured in
`benchmarks/config.yaml` through all five suites, sequentially:

1. `responsiveness` — one flow under changing BW/RTT.
2. `responsive-fairness` — 2/4/5/7 flows join evenly, then share a changing link.
3. `fairness` — two flows, flow 2 joins at 20 s.
4. `inter-rtt-fairness` — the same join test with unequal RTTs.
5. `convergence-4flow` — four flows joining at 0/25/50/75 s.

Validate everything first without starting Mininet (no root needed):

```bash
./venv_training/bin/python benchmarks/run_all.py --dry-run
```

Then run the full sweep:

```bash
sudo -E env PATH="$PATH" HOME="$HOME" ./venv_training/bin/python benchmarks/run_all.py
```

Output is one in-place progress bar per benchmark; each child runner's full
verbose output is written to `benchmarks/logs/<timestamp>_<nn>_<name>.log`.
Benchmarks resume per trial: finished cells are kept and only missing or
errored ones are re-run, so re-launching after an interruption is safe.

Useful flags:

| Flag | Effect |
| --- | --- |
| `--approach <data_folder-or-name>` | Only run this approach from `benchmarks/config.yaml`; repeatable (default: all approaches) |
| `--experiment <name>` (alias `--only`) | Only run this suite; repeatable (names as listed above) |
| `--dry-run` | Validate every selected benchmark without starting Mininet |
| `--keep-going` | Continue with later benchmarks after a failure |
| `-v` / `--verbose` | Stream raw child output instead of the progress bar |

Example — one model through two suites:

```bash
sudo -E env PATH="$PATH" HOME="$HOME" \
  ./venv_training/bin/python benchmarks/run_all.py \
    --approach ma_dreamer_20260611-175726 \
    --experiment fairness --experiment convergence-4flow
```

Results are written beneath each suite's `data/` directory. Plotting is a
separate step — see [benchmarks/README.md](benchmarks/README.md) for the
per-suite plotters and for running individual suite runners directly.

All responsive-fairness generated CSV tables and PDFs are written under
`benchmarks/benchmark_responsive_fairness/aggregate/`. The combined
`aggregate/all.pdf` compares every approach on one page per flow count.

## Benchmarking a freshly trained model

Add an entry to the `approaches:` list in `benchmarks/config.yaml` pointing at
the run's resolved config and checkpoint, e.g.:

```yaml
- data_folder: ma_dreamer_20260611-175726
  kind: model
  algorithm: ma_dreamer
  state: tempest
  reward: tempest_fairness_ma
  environment: multiflow_interleave
  plot_label: MA-Dreamer / Tempest / Team fairness
  config: olympus/models/ma_dreamer_20260611-175726/telemetry/config.resolved.yaml
  checkpoint: olympus/models/ma_dreamer_20260611-175726/checkpoints/ma_dreamer_cwnd_model.pt
```

Then run `run_all.py` as above (optionally with `--approach <data_folder>` to
benchmark only the new model).

## Benchmarking kernel congestion controls

Kernel entries need no checkpoint or learner metadata:

```yaml
- data_folder: cubic
  kind: kernel
  kernel_cc: cubic
  plot_label: CUBIC
```

The supplied benchmark config includes 14 standard kernel CCs. Load modular
algorithms with `bash kernel/ins_all.sh`, then select any subset:

```bash
sudo -E env PATH="$PATH" HOME="$HOME" \
  ./venv_training/bin/python benchmarks/run_all.py \
    --approach cubic --approach bbr
```

These runs retain and plot iperf3 JSON only; RL state, action, CWND, and reward
panels are not expected or generated.

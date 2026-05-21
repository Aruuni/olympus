# Benchmarks — architecture & how to add a new one

This document describes how the `single_agent_olympus/benchmarks/` benchmarks
are structured so a future agent can build a new one quickly. It reflects the
existing benchmarks:

- `benchmark_convergence/` — multi-flow convergence (defines the shared
  low-level helpers `_binned_series`, `_kernel_flows`, `_write_flow_plan`).
- `benchmark_responsiveness/` — random bw/RTT schedule responsiveness
  (the **shared helper library** every other benchmark imports).
- `benchmark_fairness/` — two-flow fairness heatmap (flow 2 joins late).
- `benchmark_inter_rtt_fairness/` — two-flow **inter-RTT** fairness heatmap
  (flow 1 fixed RTT, flow 2 varying RTT). Built by copying
  `benchmark_fairness` and adding per-flow RTT.
- `benchmark_convergence_4flow/` — 4 flows joining at fixed intervals
  (default 0/25/50/75 s, 100 s each, 175 s episode), **N runs per approach
  averaged**, plotted as one combined page with **one stacked panel per
  approach** of the per-flow goodput (mean line + std band, log y). The
  reference pattern for "run-averaged, multi-approach single-figure summary"
  and for handling `kernel`/`astraea`/`model` uniformly via receiver iperf.

All paths below are relative to the repo root
`/its/home/mm2350/extra/Olympusv2`.

---

## 1. Directory layout

```
single_agent_olympus/
  config.yaml                      # BASE config: paths.listener, paths.py, learner.port, ...
  orchestrator.py                  # run_episode() + runtime helpers; drives MininetEnv
  benchmarks/
    config.yaml                    # APPROACH CATALOG shared by every benchmark
    BENCHMARKS.md                  # this file
    benchmark_<name>/
      __init__.py                  # one-line docstring
      config.yaml                  # this benchmark's knobs (the `benchmark:` block)
      <name>.py                    # the runnable script
      data/                        # output_root (created on run; per-approach subdirs)
arm_bandit_mutant/
  mininet_env.py                   # MininetEnv dumbbell topology
```

A benchmark package is just: `__init__.py`, `config.yaml`, `<name>.py`. Output
goes under `data/` (the `output_root`).

---

## 2. The three config files

### Base config — `single_agent_olympus/config.yaml`
Referenced via `base_config: ../../config.yaml`. The benchmark only needs:
- `paths.listener` → `single_agent_olympus/oc_listener` (the C listener binary)
- `paths.py` → `./venv_training/bin/python`
- `learner.port` → used by `_final_runtime_cleanup` at the end of a run

### Approach catalog — `single_agent_olympus/benchmarks/config.yaml`
Referenced via `approaches_config: ../config.yaml`. One `approaches:` list shared
by all benchmarks. Each entry is one "approach" (a thing under test). Fields:

| field | meaning |
|---|---|
| `data_folder` | stable output subdir name under `data/` (falls back to `name`, then `<algorithm>_<state>_<reward>_<environment>`) |
| `kind` | `model` (default, RL checkpoint), `kernel` (in-kernel CC algo), `astraea`, or `orca` (the original SIGCOMM'20 Orca C++/TF1 impl) |
| `algorithm`, `state`, `reward`, `environment` | model identity / training env |
| `checkpoint` | path to `.pt` (model kinds only; resolved relative to the benchmark config dir) |
| `kernel_cc` | kernel CC name passed to iperf3 `-C` (kernel kinds) |
| `plot_label` | legend/title text only |

### Per-benchmark config — `benchmark_<name>/config.yaml`
```yaml
approaches_config: ../config.yaml      # shared catalog
base_config: ../../config.yaml         # base config
output_root: data                      # relative => resolved next to this file
benchmark:
  name: <unique benchmark env name>    # written into every row as benchmark_environment
  duration_s: 60
  runs_per_cell: 4
  n_parallel: 7
  bdp_mult: 4                          # INVARIANT for OC-CWND envs — do not tune
  measure_interval_s: 1.0
  kernel_cport_base: 53000
  deterministic_inference: true
  seed_offset: 0
  # ... benchmark-specific knobs (grid axes, flow schedule, score window) ...
```
A `_<name>_cfg(cfg)` function in the script reads `cfg['benchmark']` and
`setdefault`s every key, so the config file only needs the values you want to
override.

---

## 3. Approach kinds & dispatch

`_approach_slot()` (one per parallel slot) pulls work items off a queue and
dispatches on `approach['kind']`:

- `kernel` / `astraea` → `_run_kernel_trial()`: builds `MininetEnv` directly,
  runs iperf3 with `-C <kernel_cc>` (or `astraea`), no RL listener. `astraea`
  additionally starts/stops a listener via `_start_astraea_listener` /
  `_stop_astraea_listener` at the **approach** level (in `_run_approach`).
- `orca` → `_run_orca_trial()` (responsiveness: `_run_orca_baseline()`):
  builds `MininetEnv` like the kernel path but **no iperf3** — instead
  `_run_orca_on_env()` (shared, in `responsiveness.py`) launches the original
  Orca on the hosts: `orca-server-mahimahi` (data source + TF1 actor) on each
  `c{i}` and `clientThr` (sink) on each `x{i}`, via `sender.sh`/`receiver.sh`
  wrapped in `sudo -u <user>`, exactly as cctestbed runs it. No RL listener,
  no checkpoint. `_orca_receiver_to_csvs()` parses each `clientThr` capture
  (`run_dir/orca_x{f}.log`, between `----START----`/`----END----`) into the
  same `csvs/x{f}.csv` the iperf3 path produces, so all downstream
  binning/plotting/completion logic is unchanged. Per-(slot,flow) unique
  `actor_id`/port avoid SysV-IPC collisions across the shared kernel. The
  TF1 actor warmup is slack-absorbed (`orca.warmup_slack_s`) and the leading
  idle prefix trimmed (`orca.warmup_trim`). Orca prints no RTT, so a per-flow
  `ss` sidecar on the sender captures the kernel srtt of the Orca socket
  (the same `tcpi_rtt` Orca itself uses); the iperf RTT readers fall back to
  it, so `responsiveness` RTT metrics and `convergence_4flow`'s
  `efficiency.pdf` work for Orca too. Wired into `convergence_4flow`,
  `fairness`, `inter_rtt_fairness`, `responsiveness` (not the original
  `convergence`). Catalog entry: `data_folder: orca_real`. See
  [ORCA_REAL.md](ORCA_REAL.md).
- anything else → `_run_model_trial()`: resolves the checkpoint, builds a
  runtime cfg with `_prepare_runtime_cfg`, and calls
  `orchestrator.run_episode(...)` which itself constructs `MininetEnv` and
  spawns the C listener + RL worker.

The kernel/astraea/model paths end by parsing iperf3 JSON; the `orca` path
parses `clientThr` stdout. All return the same row-dict shape.

---

## 4. Shared helper library

Import infra helpers — **do not reimplement these**:

```python
from single_agent_olympus.orchestrator import (
    _as_bool, _final_runtime_cleanup, _resolve_repo_path, run_episode)
from single_agent_olympus.benchmarks.benchmark_responsiveness.responsiveness import (
    _append_csv, _approach_label_map, _approach_plot_label,
    _approach_selection_keys, _collect_approach_rows_with_labels,
    _configured_approaches, _copy_if_exists, _deterministic_env,
    _finite_float, _load_benchmark_config, _load_yaml, _parse_iperf_json,
    _prepare_runtime_cfg, _resolve_from, _restore_sudo_user_ownership,
    _safe_overwrite_dir, _safe_unlink, _slug, _start_astraea_listener,
    _stop_astraea_listener, _validate_approach, _validate_unique_data_folders,
    _write_rows, _write_schedule)
from single_agent_olympus.benchmarks.benchmark_convergence.convergence import (
    _binned_series, _kernel_flows, _write_flow_plan)
```

What the important ones do:

| helper | role |
|---|---|
| `_load_benchmark_config(path)` | load a benchmark `config.yaml`, merging `approaches_config` |
| `_configured_approaches(cfg, only)` | resolve the `approaches:` list (optionally filtered by `--approach`) |
| `_validate_approach` / `_validate_unique_data_folders` | sanity-check approach entries |
| `_approach_plot_label(a)` | human label for plots; `_approach_label_map(cfg)` for collection |
| `_resolve_from(base_dir, path)` | resolve a config-relative path; `_resolve_repo_path` for repo-relative |
| `_prepare_runtime_cfg(base_cfg, approach, ckpt, scratch, run_dir, plot_episodes=False)` | build the per-trial runtime cfg for model kinds |
| `_deterministic_env(cfg)` | force deterministic inference |
| `run_episode(cfg, ecfg, ep, listener_bin, python_bin, '', '', slot)` | run ONE model episode (builds MininetEnv + listener) |
| `_write_schedule(run_dir, schedule, bench)` | write link schedule csv → `(schedule_csv, _)` |
| `_write_flow_plan(run_dir, flow_plan)` | write `flow_schedule.csv` |
| `_parse_iperf_json(json, csv=None, measure_interval_s=)` | parse iperf3 JSON → `{goodput_mbps, samples, rtt_samples_ms}`; optionally writes a measurement csv. **RTT is only in the client/sender JSON.** |
| `_kernel_flows(run_dir, flow_plan)` | read `run_dir/csvs/x{flow}.csv` → list of per-flow `{flow,start,end,t,thr,rtt,cwnd}` |
| `_binned_series(flows, flow_plan, bench)` | 1 Hz binned aggregate: `time, per_flow, total, jain, fair_error, fair_share, active_count`. Needs `bench['duration_s']`, `bench['bw_mbps']`. |
| `_append_csv` / `_write_rows` | incremental / full CSV writers (fixed field list) |
| `_collect_approach_rows_with_labels(output_root, label_map)` | re-read all approach `metrics.csv` for `--plot-only` |
| `_safe_overwrite_dir(dir, root)` | clear an approach dir safely before a fresh run |
| `_restore_sudo_user_ownership(path)` | chown outputs back to the invoking user (runs under sudo) |
| `_final_runtime_cleanup(learner_port)` | kill stragglers at end of run |

---

## 5. Execution pipeline (per benchmark)

`main()`:
1. `_load_benchmark_config` → `_<name>_cfg` → `bench` dict.
2. Resolve `base_config`, `output_root`.
3. `--plot-only` → `_replot_all` (rebuild plots/summaries from existing
   `metrics.csv`), then return.
4. `_configured_approaches` (+ `--approach` filter) → validate.
5. `_split_by_completion`: an approach is **skipped** if its `metrics.csv`
   already has an error-free row with the expected `goodput_source` for every
   `(bw, rtt-axis, run)` cell — runs are idempotent / resumable.
6. `--dry-run` → print plan and return.
7. For each approach → `_run_approach`.
8. `finally`: `_final_runtime_cleanup`, `_restore_sudo_user_ownership`,
   then `_replot_all`.

`_run_approach`:
- `_safe_overwrite_dir`, write `run_meta.json`.
- Build the grid: for each cell × run, make `run_dir`, write the link schedule
  and `flow_schedule.csv`, push a work item onto a `multiprocessing.Queue`.
- Start `n_parallel` worker processes (`_approach_slot`), each with a unique
  `instance_id` (→ isolated mininet bridge prefix `i{id}s1/s2/s3`, iperf port
  `5201+id`, and a per-slot `cport`).
- Drain results, `_append_csv` each row, then `_write_approach_outputs`
  (per-approach `metrics.csv`, summary csv/json, heatmap pdf).

A **trial** (`_run_kernel_trial` / `_run_model_trial`):
1. Clear stale `/tmp/iperf_*` for its cport.
2. Build/run the env for `duration_s + ~3s`; iperf3 servers on receivers,
   clients on senders with per-flow `start_delays` & `flow_durations`.
3. `_copy_receiver_iperf_outputs`: copy `/tmp/iperf_server_<cport>_<f>.json`
   → `run_dir/iperf_receiver_flow<f>.json` and
   `/tmp/iperf_<cport>_<f>.json` → `run_dir/iperf_client_flow<f>.json`,
   parsing receivers into `run_dir/csvs/x<f>.csv`.
4. `_binned_series` → score-window metrics + goodput-ratio stats.
5. Plot the run PDF, return the row dict.

---

## 6. Key data structures

- **flow_plan**: `[{'flow':1,'start':0.0,'duration':D,'end':D}, {'flow':2,'start':S,'duration':D-S,'end':D}]`.
  Drives iperf `start_delays`/`flow_durations` and the score-window/active-flow logic.
- **schedule** (for `_write_schedule`): `{'seed','initial':{'bw','delay'},'changes':[],'rows':[{'t','bw','delay'}]}`.
  Static benchmarks use a single row (no `changes`).
- **iperf JSON files in each `run_dir`**: `iperf_receiver_flow<f>.json`
  (goodput; → `csvs/x<f>.csv`) and `iperf_client_flow<f>.json`
  (**has TCP RTT**: `intervals[].streams[].rtt` µs → `rtt_samples_ms`).
- **row dict**: flat dict keyed by the benchmark's `FIELDS` list; written by
  `_append_csv`/`_write_rows`. Always include `approach, run_dir, error,
  goodput_source, bw_mbps`, and the grid axis columns; unknown fields are
  blanked via `row.setdefault`.
- **score window**: last `score_window_s` of the episode where >0 flows are
  active; metrics (goodput ratio etc.) are computed only there.

---

## 7. MininetEnv (`arm_bandit_mutant/mininet_env.py`)

Dumbbell: `c1..cn → s1 --[delay]--> s2 --[bw/queue]--> s3 → x1..xn`. The
propagation delay is on the **shared** `s1–s2` link, so by default every flow
has the same RTT. Bottleneck bw/queue is on `s2–s3`; `qsize = bdp_mult * BDP`.

Key ctor args: `n, bw, delay, bdp_mult, duration, cport, cc_algo,
instance_id, unique_cports, per_flow_delays`.

`per_flow_delays` (added for inter-RTT fairness): optional list of the
**full one-way delay (ms) per flow**, applied as netem on each sender's access
link `c_i → s1`. **When set, the shared `s1–s2` link carries no netem at all** —
each flow's RTT is determined solely by its own access link, and `delay` is
then used *only* to size the bottleneck BDP buffer (`qsize`). `None` ⇒ legacy
behaviour: `delay` on the shared `s1–s2` link (every existing benchmark passes
nothing and is unaffected). For inter-RTT:
`per_flow_delays = [rtt1, rtt2]`, `delay = min(rtt1, rtt2)` (buffer sizing
only). For model kinds, pass it through `ecfg['per_flow_delays']` (orchestrator
forwards it to `MininetEnv`); for kernel kinds, pass it to the `MininetEnv(...)`
ctor directly.

---

## 8. CLI surface

Every benchmark script exposes the same flags via `main()`:

```
--config <path>        # benchmark config.yaml (default: alongside the script)
--approach NAME         # repeatable; restrict to these data_folders/names
--plot-only             # rebuild plots/summaries from existing data, no runs
--heatmaps-only         # with --plot-only: skip per-run PDFs
--dry-run               # print the plan only
```

Run (needs root for mininet/tc; preserve env):

```
sudo -E env PATH="$PATH" HOME="$HOME" ./venv_training/bin/python \
  single_agent_olympus/benchmarks/benchmark_<name>/<name>.py \
  --config single_agent_olympus/benchmarks/benchmark_<name>/config.yaml
```

---

## 9. Recipe: add a new benchmark

1. `mkdir benchmark_<name>`; add `__init__.py` (one-line docstring).
2. Copy the closest existing `<name>.py` (fairness for heatmaps,
   responsiveness for single-flow schedules, convergence for many flows).
   Reuse the shared imports in §4 verbatim.
3. Write `config.yaml`: `approaches_config: ../config.yaml`,
   `base_config: ../../config.yaml`, `output_root: data`, and a `benchmark:`
   block. Add a `_<name>_cfg(cfg)` that `setdefault`s every knob.
4. Define the variation surface:
   - the grid (`_grid_cells`), the flow schedule (`_flow_plan`), and the
     link schedule (`_static_schedule` / a dynamic one).
   - what the **reported metric** is and its `goodput_source` constant
     (used by the completion check, so make it unique per metric semantics).
5. Wire `_run_kernel_trial` and `_run_model_trial` to your env settings
   (bw/delay/`per_flow_delays`/duration/flow_plan). Keep both paths — kernel
   approaches (cubic/bbr/astraea) are usually part of the comparison.
6. Define `FIELDS` (row) and `SUMMARY_FIELDS`, plus `_row`, `_summary_rows`,
   `_plot_*`. Keep `error`, `goodput_source`, `run_dir`, `bw_mbps`, grid axes.
7. Keep `main()`'s structure (plot-only / dry-run / completion-skip /
   sudo-ownership restore / final cleanup) — copy it; only the bench-specific
   bits change.
8. Validate without root: `py_compile` the script, then run `--dry-run`
   (loads configs, resolves approaches, prints the grid size). For plotting
   code, fabricate `run_dir/csvs/x<f>.csv` (`time,bandwidth` columns) and call
   the plot fns directly — no mininet/root needed.

### Run-averaged, single-figure summary (the `benchmark_convergence_4flow` pattern)

When the deliverable is "one page, one panel per approach, runs averaged"
(Fig.13-style):

- The parallel work items are **runs** (`run1..runN`), not grid cells; one
  approach per data folder, `runN` subdirs inside it.
- Treat every kind the same: run iperf with `-C <cc>` / `astraea` / via
  `run_episode`, then take **receiver iperf goodput** (`_copy_receiver_iperf_
  outputs` → `_kernel_flows` → `_binned_series`). Avoids model-vs-kernel
  goodput-source skew. `_binned_series` is fixed-length (`duration_s`) so runs
  stack cleanly.
- Average: `np.stack` per-run `series['per_flow']` → `(runs, n_flows, T)`,
  then `np.nanmean`/`np.nanstd` over axis 0 inside
  `warnings.catch_warnings()` (inactive bins are all-NaN and warn otherwise).
- Summary = `plt.subplots(n_approaches, 1, sharex=True, squeeze=False)`, one
  panel per approach, per-flow mean line + `fill_between(mean±std)`, optional
  `set_yscale('log')` (mask ≤0 first), vlines at join/leave times, written as
  one PDF via a `*.tmp.pid` + `os.replace` swap.

---

## 10. Invariants & gotchas

- **`bdp_mult` is fixed at 4** for OC-CWND environments — it is an environment
  invariant, not a tuning knob. Do not change it.
- Runs are **idempotent**: completed approaches are skipped via the
  `metrics.csv` completion check, so a benchmark can be re-run to finish a
  partial sweep. The check keys on `(bw, rtt-axis, run)` + matching
  `goodput_source` + empty `error` — keep those columns correct.
- Scripts run under **sudo**; always finish with
  `_restore_sudo_user_ownership(output_root)` or outputs end up root-owned.
- Each parallel slot must use isolated resources: distinct `instance_id`
  (mininet prefix + iperf port) and a distinct `cport`
  (`<cport_base> + instance_id*100`). Never share cports across slots unless
  `unique_cports`/single-listener semantics require it.
- iperf3 reports **TCP RTT only in the client/sender JSON**
  (`iperf_client_flow<f>.json`). Receiver JSON has goodput, not RTT.
- `_binned_series` needs `bench['duration_s']` and `bench['bw_mbps']` in the
  small dict you pass it (not the full benchmark config).
- The `benchmark.name` string is recorded as `benchmark_environment` in every
  row — make it unique per benchmark so collected CSVs don't collide.
```

# Real Orca (`kind: orca`) benchmark integration

This documents the integration of the **original SIGCOMM'20 Orca** C++/TF1
implementation (the `Aruuni/Orca` repo as installed for cctestbed) into the
`single_agent_olympus` benchmark suite as a first-class baseline approach.

> This is **not** the `orca` PyTorch re-creation (see
> [ORCA_ALIGNMENT.md](../ORCA_ALIGNMENT.md) for that). This is the real
> compiled Orca server + its TensorFlow-1 actor, launched on the mininet
> hosts exactly as cctestbed launches it.

All paths are relative to the repo root `/its/home/mm2350/extra/Olympusv2`.

---

## 1. What it is

Orca does **not** use iperf3. It ships its own transport pair:

| Binary | Runs on | Role |
|---|---|---|
| `rl-module/orca-server-mahimahi` | sender `c{i}` | data **source**; spawns a TF1 actor (`d5.py`) that steers cwnd |
| `rl-module/clientThr` | receiver `x{i}` | data **sink**; prints per-interval goodput |

`kind: orca` is dispatched like `kernel`/`astraea` (builds `MininetEnv`
directly, no RL listener, no checkpoint) but instead of `env.run_iperf()` it
launches Orca's own `sender.sh`/`receiver.sh` on the hosts — wrapped in
`sudo -u <user>`, byte-for-byte the way `core/emulation.py`
`start_orca_sender`/`start_orca_receiver` does in cctestbed.

`clientThr`'s per-interval throughput is parsed into the **same**
`run_dir/csvs/x{f}.csv` (`time,bandwidth` in Mbps) that the iperf3 path
produces, so every downstream step — binning, metrics, plots, the
completion/idempotency check — is unchanged.

---

## 2. Data path

```
c{i}:  sudo -u <user> env HOME=<home> EXPERIMENT_PATH=<run_dir> \
         bash <install>/sender.sh   <port> <actor_id> <finish> <install>
         └─ orca-server-mahimahi <port> <install>/rl-module 100 cubic <actor_id> <finish> 0
            └─ $HOME/venv/bin/python d5.py ... --task=<actor_id>  (TF1 actor)
         stdout → run_dir/orca_c{f}.log

x{i}:  sudo -u <user> env HOME=<home> \
         bash <install>/receiver.sh <c{i}.IP()> <port> 0 <install>
         └─ clientThr <ip> 0 <port> 1
         stdout → run_dir/orca_x{f}.log
                  └─ between ----START----/----END----:
                     tick , <mbps>e+06 , total_bytes , avg_mbps
```

Parsing (`_parse_orca_client_output`, mirrors cctestbed `parse_orca_output`):
field 2 is Mbps with a literal `e+06` suffix, so `float()` yields `Mbps*1e6`
and we divide by `1e6` back to Mbps. RTT is **not** in either stream
(see §6). Output is written via the shared `_write_measurement_csv`, so the
csv is identical in shape to the iperf3 one.

---

## 3. Files changed

| File | Change |
|---|---|
| [benchmark_responsiveness/responsiveness.py](benchmark_responsiveness/responsiveness.py) | **Shared helpers** (`_ORCA_DEFAULTS`, `_orca_settings`, `_orca_run_user`, `_orca_user_home`, `_orca_actor_id`, `_orca_port`, `_orca_paths`, `_run_orca_on_env`, `_parse_orca_client_output`, `_orca_receiver_to_csvs`); `_validate_approach` accepts `kind: orca`; `_run_orca_baseline` + slot dispatch + checkpoint-skip |
| [benchmark_convergence_4flow/convergence_4flow.py](benchmark_convergence_4flow/convergence_4flow.py) | imports + `_run_orca_trial` + `_approach_slot` dispatch + checkpoint-skip |
| [benchmark_fairness/fairness.py](benchmark_fairness/fairness.py) | imports + `_run_orca_trial` + dispatch + checkpoint-skip |
| [benchmark_inter_rtt_fairness/inter_rtt_fairness.py](benchmark_inter_rtt_fairness/inter_rtt_fairness.py) | imports + `_run_orca_trial` (with `per_flow_delays`) + dispatch + checkpoint-skip |
| [benchmarks/config.yaml](config.yaml) | catalog entry `data_folder: orca_real` + `orca:` block |
| [benchmarks/BENCHMARKS.md](BENCHMARKS.md) | kind table + dispatch section updated |

The shared helpers live in `responsiveness.py` (the suite's shared library,
alongside the `_*_astraea_*` helpers) and are imported by the other three
benchmarks — the same pattern astraea uses. The original `benchmark_convergence`
was intentionally **not** wired (per scope).

---

## 4. Catalog entry & config knobs

```yaml
- data_folder: orca_real
  kind: orca
  algorithm: orca
  state: kernel
  reward: none
  environment: dynamic
  plot_label: Orca (original)
  orca:
    install_dir: /its/home/mm2350/Desktop/mininettestbed/CC/Orca
    port_base: 47000
    warmup_slack_s: 60
    server_lead_s: 1.0
    warmup_trim: true
    warmup_trim_eps_mbps: 0.05
```

| Knob | Default | Meaning |
|---|---|---|
| `install_dir` | `…/Desktop/mininettestbed/CC/Orca` | Orca repo; must contain `sender.sh`, `receiver.sh`, `rl-module/{orca-server-mahimahi,clientThr}` |
| `port_base` | `47000` | per-flow port = `port_base + slot*100 + flow` |
| `warmup_slack_s` | `60` | extra wall time so the TF1 actor can load before the duration timer starts (also settable per-benchmark as `orca_warmup_slack_s`) |
| `server_lead_s` | `1.0` | start the server this long before its `clientThr` |
| `warmup_trim` | `true` | drop the leading idle/near-zero prefix and rebase the time axis |
| `warmup_trim_eps_mbps` | `0.05` | "active" threshold for the trim |

`_metric_row` requires `algorithm/state/reward/environment`, so those four
keys are present (mirrors the astraea entry); `checkpoint` is **not** required
(`_validate_approach` treats `orca` like `astraea`).

---

## 5. Concurrency & parallel-emulation safety

Benchmarks run `n_parallel` slots, each its own `MininetEnv` (`instance_id`),
all on **one host kernel**. mininet isolates *network* namespaces only —
**not** IPC namespaces — so the safety analysis is per resource:

| Resource | Namespace-isolated? | How parallel safety is guaranteed |
|---|---|---|
| TCP ports / connectivity | yes (per-slot netns + bridges `i{id}s1/2/3`) | each slot's hosts are a separate netns; `c.IP()` is in-namespace |
| `ss` poller | yes | runs via `c{i}.cmd()` **inside that host's netns**, so it only ever sees that one flow's socket — a parallel slot's `ss` cannot see or perturb it, even with identical ports |
| **SysV-IPC shmem** (Orca server↔actor) | **NO — kernel-global** | `key = actor_id*10000 + rand()%10000 + 1`; `actor_id = instance_id*100 + flow` gives every (slot,flow) a **disjoint 10 000-wide key range** (slot0/f1→keys 10001‑20000, slot1/f1→1010001‑1020000, …) |
| port | n/a (belt-and-suspenders) | `port = port_base + instance_id*100 + flow`, unique per (slot,flow) |
| TF event dir | n/a | per `actor_id` (`job_name+task`), disjoint |
| output files | n/a | `run_dir` is per (approach,cell,run); `orca_*_c{f}.log` per flow |

The SysV-IPC range is **the** critical one (it's the only Orca resource not
covered by netns isolation) and it scales: at `n_parallel=50`, flow 1 →
`actor_id=4901` → keys ≈ 49 020 000, still disjoint, well within `key_t`.

The `ss` sidecar inherits netns isolation automatically (it's a `c{i}.cmd()`),
and additionally filters `state established '( sport = :<port> )'` so within a
host it ignores the LISTEN socket and the actor (which uses SysV shmem, not
TCP). Flows self-terminate (server exits after `finish`, `clientThr` flushes
& exits; `ss` is `timeout`-bounded and also dies with the netns at
`env.stop()`), so teardown is just `env.stop()`. **No cross-slot interference
on any resource.**

---

## 6. RTT via the `ss` sidecar (implemented)

Neither Orca stream prints RTT (`clientThr` → throughput only;
`orca-server-mahimahi` → `timestamp,pacing_rate,bytes_sent`). So, exactly as
cctestbed does with `ss_script.sh`, `_run_orca_on_env` starts a per-flow `ss`
poller on the **sender** `c{i}` for the kernel srtt of the Orca socket:

```
timeout <dur+slack> bash -c \
  'while :; do ss -tinHO state established "( sport = :<port> )" \
     | ts "%.s,"; sleep <ss_interval_s>; done'  > run_dir/orca_ss_c{f}.log
```

`ss -i` reports `rtt:<srtt>/<rttvar>` (ms). That srtt is the **same**
`tcpi_rtt` Orca itself reads via `getsockopt(TCP_INFO)` for its RL state — it
is the true kernel srtt, not an approximation. Parsing
(`_parse_orca_ss` → `_orca_ss_rtt_samples` / `_orca_ss_rtt_flows`): pull the
`rtt:` token, rebase time to the first sample (the socket only exists once
the flow connects — aligns with the goodput clock like `warmup_trim`), add
the flow's start.

Wired so existing call sites need no change — the iperf RTT readers fall
back to ss when no iperf client JSON is present:

- `responsiveness`: `mean_rtt_ms` / `p95_rtt_ms` populated from ss
- `convergence_4flow`: `_iperf_rtt_samples` → ss fallback ⇒ **`efficiency.pdf`
  (norm-delay vs norm-goodput) now has Orca points**
- `inter_rtt_fairness`: `_iperf_rtt_flows` → ss fallback ⇒ the per-flow RTT
  plot panel renders Orca's srtt trace
- `fairness`: unaffected (no RTT/efficiency); ss logs still written, harmless

Controlled by `orca.ss_enabled` (default `true`) and `orca.ss_interval_s`
(default `0.1`, cctestbed's value). Set `ss_enabled: false` to skip it
(RTT/efficiency then revert to NaN/empty for Orca).

> **Re-run note:** the idempotency check skips approaches whose `metrics.csv`
> is already complete. An `orca_real` folder produced **before** this `ss`
> work has no `orca_ss_*.log`, so its efficiency points stay empty — delete
> that approach's data dir (e.g. `…/data/orca_real`) to regenerate with RTT.

---

## 7. Warmup slack & trim

Orca's per-flow cold start is heavy: `orca-server-mahimahi` spawns
`$HOME/venv/bin/python d5.py` (TF1), which imports TensorFlow and loads the
trained checkpoint, and the server **blocks until the actor signals ready**
before starting its data thread *and* its `duration` timer. So real wall time
≈ `actor_warmup + duration`.

- `warmup_slack_s` is added to the single episode sleep
  (`sleep(duration_s + warmup_slack_s)`) so the run does not tear the
  namespace down mid-transfer. It is sized to the **warmup**, not idle
  padding — work (model load → data) happens during it; only the leftover
  after the flow finishes is genuinely idle. Lower it for single fast flows;
  raise it (or `orca_warmup_slack_s`) if `orca_x{f}.log` captures look
  truncated under heavy parallelism.
- `warmup_trim` then removes the leading near-zero prefix (the period before
  the actor was ready) and rebases the time axis, so Orca's *active* window
  aligns with the fixed benchmark window like every other CC. This strips a
  process/model-load latency, not Orca's congestion-control behaviour.

Observed warmup in a real `bw50_rtt20/run1` is ~2.5 s (first sample at
`t≈2.54`), but it grows with the number of TF1 actors loading at once.

---

## 8. Running it

Per benchmark (needs root for mininet/tc; preserve env):

```bash
# 4-flow convergence
sudo -E env PATH="$PATH" HOME="$HOME" ./venv_training/bin/python \
  single_agent_olympus/benchmarks/benchmark_convergence_4flow/convergence_4flow.py \
  --config single_agent_olympus/benchmarks/benchmark_convergence_4flow/config.yaml \
  --approach orca_real

# two-flow fairness heatmap
sudo -E env PATH="$PATH" HOME="$HOME" ./venv_training/bin/python \
  single_agent_olympus/benchmarks/benchmark_fairness/fairness.py \
  --config single_agent_olympus/benchmarks/benchmark_fairness/config.yaml \
  --approach orca_real

# inter-RTT fairness heatmap
sudo -E env PATH="$PATH" HOME="$HOME" ./venv_training/bin/python \
  single_agent_olympus/benchmarks/benchmark_inter_rtt_fairness/inter_rtt_fairness.py \
  --config single_agent_olympus/benchmarks/benchmark_inter_rtt_fairness/config.yaml \
  --approach orca_real

# responsiveness (random bw/RTT schedule)
sudo -E env PATH="$PATH" HOME="$HOME" ./venv_training/bin/python \
  single_agent_olympus/benchmarks/benchmark_responsiveness/responsiveness.py \
  --config single_agent_olympus/benchmarks/benchmark_responsiveness/config.yaml \
  --approach orca_real
```

Drop `--approach orca_real` to run it alongside the other catalogued
approaches. `--dry-run` validates config/resolution without mininet.

Per-run artifacts in each `run_dir`: `orca_c{f}.log` (server stdout),
`orca_x{f}.log` (clientThr stdout), `csvs/x{f}.csv` (parsed goodput, the
common interface), plus the usual per-run/summary PDFs and `metrics.csv`.

---

## 9. Operational caveats

- **Heavy.** One TF1 Python actor *per concurrent flow*. `convergence_4flow`
  at `n_parallel: 6` × 4 flows = up to 24 actors + 24 servers loading
  TensorFlow simultaneously; `responsiveness` defaults to `n_parallel: 50`.
  Lower the benchmark's `n_parallel` if the host can't sustain
  `slots × flows` actors (symptom: truncated/empty `orca_x{f}.log`,
  "missing Orca receiver goodput samples" errors).
- **`$HOME/venv`.** `orca-server-mahimahi` hard-codes
  `getenv("HOME")/venv/bin/python`; the launch passes `env HOME=<user home>`
  so it resolves to the Orca TF1 venv (`/its/home/<user>/venv`, py3.7).
- **`sudo -u <user>`** comes from `SUDO_USER` (benchmarks run under `sudo`),
  else the current user — matching cctestbed and keeping Orca's
  `rl-module/{log,train_dir}` writes user-owned.
- **`goodput_source`** is left as the benchmarks' existing constant so the
  idempotency/completion check still skips finished `orca_real` cells; the
  `kind`/`algorithm` columns identify Orca rows.

---

## 10. Validation status

- `py_compile` — all four benchmark scripts ✓ (incl. ss wiring)
- `--dry-run` — all four resolve `orca_real`, pass `_validate_approach`
  (no checkpoint), and emit the expected grid ✓
- clientThr parser unit test — `e+06`→Mbps, warmup-prefix trim + time
  rebase, csv shape, safe fallbacks on missing/markerless logs ✓
- ss parser unit test — `rtt:` token extraction, time rebase,
  episode-clock + per-flow-shape (`_iperf_rtt_flows`-compatible) outputs,
  safe fallbacks on missing/token-less lines ✓
- ss command composition — exact nested-quoted
  `timeout … bash -c "while…ss -tinHO…|ts '%.s,'…"` verified to run,
  `timeout`-bound, and emit parseable `<epoch>,…rtt:…` lines ✓
- Parallel-emulation safety — reviewed per resource (§5): netns isolates
  the `ss` poller and connectivity; disjoint per-(slot,flow) `actor_id`
  keeps the kernel-global SysV-IPC keys non-overlapping ✓
- Live `bw50_rtt20/run1` (convergence_4flow) — actor loaded, `----START----`
  reached, ~100 s of data captured in `orca_c1.log` ✓ (pre-ss run; delete
  the folder to regenerate with RTT — see §6)

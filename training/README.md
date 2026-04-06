# Olympus v2 — Training Stack

Adaptive TCP congestion control via Option-Critic reinforcement learning.
The system trains a neural network to select which CC algorithm (arm) to run
and for how long (dwell), by observing live TCP socket statistics and switching
the kernel's CC module via the `mutant` driver.

---

## Quick start

```bash
# Clean up any leftover Mininet state
sudo mn -c

# Run training (loops forever through the experiments list)
sudo -E env PATH="$PATH" HOME="$HOME" \
  python training/orchestrator.py \
    --config   training/test_full_config.yaml \
    --listener ./oc_listener \
    --py       "$(pwd)/venv_training/bin/python"
```

Training accumulates across runs. Delete `training/oc_model_test.pt` to start fresh.

---

## System overview

```
┌─────────────────────────────────────────────────────────────┐
│  orchestrator.py  (runs as root, one episode at a time)     │
│                                                             │
│   1. Build Mininet topology  (mininet_env.py)               │
│   2. Start oc_listener binary                               │
│   3. Run iperf3 for `duration` seconds                      │
│   4. Kill listener, tear down Mininet                       │
│   5. Repeat with next environment config                    │
└──────────────┬──────────────────────────────────────────────┘
               │ spawns
               ▼
┌─────────────────────────────────────────────────────────────┐
│  oc_listener  (C binary)                                    │
│                                                             │
│  • Scans all network namespaces every scan_ms (100 ms)      │
│  • Finds ESTAB TCP connections whose source port == cport   │
│  • For each new flow:                                       │
│      – Duplicates the socket fd (pidfd_getfd)               │
│      – Sets CC to "mutant" via setsockopt                   │
│      – Opens a netlink socket to mutant kernel module       │
│      – Creates two OS pipes (O_CLOEXEC)                     │
│      – fork + exec → oc_worker.py  (the Python worker)     │
│      – Runs a per-flow C thread that:                       │
│          · Reads TCP_INFO every scan_ms → writes to pipe    │
│          · Reads arm actions from pipe → switches mutant arm│
│          · Re-enables DeepCC after every arm switch         │
└──────────────┬──────────────────────────────────────────────┘
               │ fork + exec (one process per flow)
               ▼
┌─────────────────────────────────────────────────────────────┐
│  oc_worker.py  (Python, one process per TCP flow)           │
│                                                             │
│  Combined actor + learner — no separate process needed.     │
│                                                             │
│  Startup:                                                   │
│    • Load checkpoint from OC_CHECKPOINT (if it exists)      │
│    • model weights + optimizer state + global step counter  │
│                                                             │
│  Per dwell period:                                          │
│    1. Read oc_state_t frames from state pipe (blocks)       │
│    2. Normalise → float32 state vector (9 dims)             │
│    3. Accumulate reward = log(tput_Mbps) - log(rtt_ms)      │
│    4. When dwell timer expires:                             │
│         a. Forward pass → sample (arm, dwell_ms)           │
│         b. Write oc_action_t to action pipe → C switches arm│
│         c. Store Experience in local replay buffer          │
│         d. If buffer >= min_replay: gradient step           │
│         e. Print training metrics + append to CSV log       │
│                                                             │
│  On exit (pipe closed by C when flow ends):                 │
│    • Push terminal experience (done=True)                   │
│    • Save checkpoint                                        │
└─────────────────────────────────────────────────────────────┘
```

---

## Pipe protocol

Two OS pipes connect the C flow thread to the Python worker:

```
C thread  ──state_pipe──►  Python worker
C thread  ◄─action_pipe──  Python worker
```

**state_pipe** (C writes, Python reads) — sent every `scan_ms`:
```c
// oc_state_t  44 bytes  '<8IQ'
struct {
    uint32_t cur_arm;        // currently active mutant arm ID
    uint32_t rtt_us;         // smoothed RTT (µs)        tcpi_rtt
    uint32_t rttvar_us;      // RTT variance (µs)        tcpi_rttvar
    uint32_t min_rtt_us;     // minimum RTT seen (µs)    tcpi_min_rtt
    uint32_t snd_cwnd;       // congestion window (MSS)  tcpi_snd_cwnd
    uint32_t lost;           // unrecovered lost pkts     tcpi_lost
    uint32_t retrans;        // retransmitted pkts        tcpi_retrans
    uint32_t delivered;      // cumulative delivered      tcpi_delivered
    uint64_t delivery_rate;  // bytes/s                  tcpi_delivery_rate
};
```

**action_pipe** (Python writes, C reads) — sent once per dwell period:
```c
// oc_action_t  8 bytes  '<II'
struct {
    uint32_t arm_id;    // mutant arm to switch to
    uint32_t dwell_ms;  // informational only (C ignores; Python times it)
};
```

---

## State space (STATE_DIM = 9)

All features are `log1p`-scaled to compress TCP's wide dynamic range.

| index | field | source | normalisation |
|-------|-------|--------|---------------|
| 0 | smoothed RTT (µs) | `tcpi_rtt` | log1p / 13 |
| 1 | RTT variance (µs) | `tcpi_rttvar` | log1p / 10 |
| 2 | min RTT (µs) | `tcpi_min_rtt` | log1p / 13 |
| 3 | CWND (MSS units) | `tcpi_snd_cwnd` | log1p / 7 |
| 4 | lost packets | `tcpi_lost` | log1p / 5 |
| 5 | retransmitted packets | `tcpi_retrans` | log1p / 5 |
| 6 | delivered (cumulative) | `tcpi_delivered` | log1p / 20 |
| 7 | delivery rate (bytes/s) | `tcpi_delivery_rate` | log1p / 20 |
| 8 | current arm (normalised) | mutant arm_id | arm_idx / 4 → [0, 1] |

---

## Action space — the arms (N_OPTIONS = 5)

| model index | mutant arm_id | CC algorithm |
|-------------|---------------|--------------|
| 0 | 0 | CUBIC |
| 1 | 2 | BBR v1 |
| 2 | 5 | Vegas |
| 3 | 12 | BBR v3 |
| 4 | 13 | Astraea |

The model picks both **which arm** and **how long to stay on it** (dwell time).

---

## Reward

PCC-style log utility, computed per state frame and averaged over the dwell:

```
r = log(throughput_Mbps + ε) − α · log(rtt_ms + ε)
```

`α = 1.0` (equal weight on throughput and latency).
Maximising this drives the agent to find arms that are fast without inflating RTT.

---

## Model — OptionCriticNet (model.py)

A shared-encoder network with four heads:

```
state (9,) → Linear(9,64) → ReLU → Linear(64,64) → ReLU → h (64,)
                                                              │
                    ┌─────────────────────────────────────────┤
                    │                                         │
              option_head                               value_head
              π(ω|s) (5,)                            Q(s,ω)  (5,)
              softmax                                         │
                    │                                    term_head
              dwell_mu (5,)                            β(s,ω)  (5,)
              dwell_logsig (5,)                           sigmoid
```

**At inference** (`net.act(state)`):
1. Sample option ω from π(ω|s) (stochastic)
2. Sample dwell T ~ LogNormal(μ_ω, exp(σ_ω)), clamped to [500ms, 30000ms]
3. Return (arm_id, T)

**Dwell initialisation**: `dwell_mu` bias is initialised to `log(2000)` so the
initial median dwell is 2000 ms — matching the worker's initial dwell and
keeping it away from the 500 ms minimum clamp from the start.

---

## Loss (model.py — compute_loss)

Given a batch of `Experience(s, ω, R, s', done)` tuples:

```
Advantage   A = R + γ · max_{ω'} Q(s',ω') · (1-done) − Q(s,ω)

L_value   = MSE( Q(s,ω),  target )          TD(0) critic
L_policy  = −log π(ω|s) · A.detach()        policy gradient
L_entropy = −H(π(·|s))                      exploration bonus
L_term    = β_ω(s') · (Q(s',ω) − max Q(s',·) + margin).detach()
L_dwell   = −clamp(log p(T|μ_ω,σ_ω), −10, 10) · A.detach()

Total = L_value + L_policy
      + c_entropy · L_entropy
      + c_term    · L_term
      + c_dwell   · L_dwell
```

The `log_prob` in `L_dwell` is clamped to ±10 to prevent gradient explosion
during early training when the distribution parameters are random.

### What the CSV columns mean

`step, loss, value, policy, entropy, term, dwell, cport, flow_id`

| column | healthy range | what it means |
|--------|--------------|---------------|
| `loss` | decreasing toward 0 | total loss (can be negative if entropy/dwell dominate) |
| `value` | positive, shrinking | TD error — how wrong the critic is |
| `policy` | small negative | exploiting good actions |
| `entropy` | 1.0–1.6 early, gradual ↓ | exploration; drops as policy concentrates |
| `term` | small, ±0.5 | termination signal |
| `dwell` | ±10 (clamped) | REINFORCE on dwell time; negative = avoiding bad dwells |
| `cport` | 20000 | experiment identifier (not a reward) |
| `flow_id` | 1 | flow within the experiment |

---

## Mininet topology (mininet_env.py)

Dumbbell topology — one bottleneck link:

```
c1 ──┐                         ┌── x1
     ├── s1 ──[delay]── s2 ──[bw/queue]── s3 ─┤
cN ──┘                         └── xN
```

- **s1–s2**: propagation delay + optional loss (netem)
- **s2–s3**: bandwidth + queue (tbf)
- No controller — OVS runs in `failMode='standalone'`
- `autoSetMacs` + `autoStaticArp` eliminates ARP/MAC learning delays

iperf3 clients use `-C mutant` so the kernel uses the mutant CC driver from
the start, and `--cport=N` so oc_listener can identify which flow belongs to
this experiment.

---

## Config file reference

```yaml
training:
  checkpoint:   training/oc_model.pt  # shared across all episodes
  batch_size:   32                    # experiences per gradient step
  min_replay:   32                    # buffer size before training starts
  lr:           3e-4                  # Adam learning rate
  gamma:        0.99                  # TD discount factor
  save_every:   10                    # checkpoint every N gradient steps
  c_entropy:    0.05                  # exploration pressure (raise if policy collapses)
  c_term:       0.01                  # termination gradient weight
  c_dwell:      0.001                 # dwell REINFORCE weight (keep small early)

scan_ms:    100        # how often oc_listener polls TCP_INFO (ms)
cport_base: 20000      # iperf3 client source port for flow identification

experiments:           # cycled sequentially: episode 0→env0, episode 1→env1, ...
  - {bw: 10,  delay: 20, bdp_mult: 1.0, flows: 1, duration: 60}
  - {bw: 50,  delay: 30, bdp_mult: 2.0, flows: 1, duration: 60}
  - {bw: 100, delay: 10, bdp_mult: 1.5, flows: 1, duration: 60}
```

---

## File map

| file | role |
|------|------|
| `orchestrator.py` | Top-level loop: Mininet + listener lifecycle |
| `mininet_env.py` | Mininet topology, tc qdiscs, iperf3 launch |
| `oc_worker.py` | Per-flow actor + inline learner |
| `model.py` | OptionCriticNet, Experience namedtuple, compute_loss |
| `test_full_config.yaml` | Config for smoke-testing |
| `../oc_listener.c` | C binary: flow detection, pipe protocol, mutant switching |

---

## Hyperparameter tuning notes

**Policy collapse (always picks one arm):**
Increase `c_entropy`. The entropy loss is the only force keeping the policy
from collapsing once one arm accumulates slightly higher Q-values.

**Dwell stuck at 500 ms (minimum clamp):**
The `dwell_mu` head was initialised to output ≈ 0, giving LogNormal median ≈ 1 ms.
This is fixed by the `nn.init.constant_(dwell_mu.bias, log(2000))` in model.py.
If it regresses, delete the checkpoint and restart.

**Dwell loss exploding (values like −1000):**
`c_dwell` is too large relative to how random the network is. Reduce it, or
the ±10 clamp on `log_prob` in `compute_loss` will contain it automatically.

**Training not starting (no `[learner] step=` lines):**
`min_replay` experiences haven't accumulated yet. Each dwell period (~2 s)
produces one experience. With `min_replay=32` training starts after ~64 s.

**Multiple workers spawning per flow:**
Rebuild `oc_listener`: `cc -O2 -Wall -Wextra -pthread -o oc_listener oc_listener.c`
The binary must be newer than the source or the `mutant_begin < 0` fix won't be active.

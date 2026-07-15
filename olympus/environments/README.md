# Environment backends

A backend turns an episode's link/traffic spec into flows the
congestion-control workers attach to. Backends live under
`olympus/environments/<type>/env.py`, expose an `ENV_CLASS` implementing
`NetworkEnv` ([base.py](base.py)), and are selected by the environment `type`
in config via `make_env`. Copy [`_template/`](_template/env.py) to start a new
one.

## The contract

Callers (orchestrator, benchmarks) drive exactly five methods, in order:

| Phase | Method | Meaning |
|---|---|---|
| 1 | `start()` | bring up backend resources (topology / namespaces / services) |
| 2 | `setup_environment(link_schedule=None)` | apply the initial link scenario and take ownership of the mid-episode change schedule |
| 3 | `start_episode(monitor_interval, start_delays, flow_durations, episode_start)` | launch the flows; returns immediately |
| 4 | `wait()` | block until the episode has finished; raise on failure |
| 5 | `stop()` | tear down everything `start()` created |

`change_link(bw, delay, loss)` is **optional**: the base-class default accepts
and ignores the call. Override it only if the backend can retune the
bottleneck in place.

## Scenario ownership

The scenario is declarative from the caller's side. The static link comes in
through the constructor kwargs (documented in [base.py](base.py)); the
mid-episode schedule — a list of `{'t': seconds, 'bw': ..., 'delay': ...,
'loss': ...}` entries — is handed to `setup_environment`. The backend chooses
how to realize it:

- **Live replay** (Mininet): override `change_link`, then call
  `self._begin_link_schedule(episode_start)` at the end of `start_episode`
  and `self._stop_link_schedule()` at the top of `stop`. The base class runs
  the replay thread and drives `change_link` at each entry's offset.
- **Internal execution** (RayNet): consume the schedule in
  `setup_environment` (RayNet folds it into the episode config the simulator
  executes) and never start the replay.

**Timing invariant:** `episode_start` (a `time.monotonic` value) is the same
anchor the orchestrator exports to workers as `OC_EPISODE_START`. Reward and
state plugins locate the current link on the schedule from that anchor, so a
live replay must be anchored there too — anchoring anywhere else shifts the
executed schedule relative to the timeline the workers assume.

## Flow identity

Flow `i` is addressed by the key `cport` (or `cport + i - 1` with
`unique_cports`). Workers, listeners, and parsers use that key to find their
flow; what it binds to is the backend's bucket:

- **Emulation** (real kernel TCP flows — Mininet): the key *is* the iperf3
  client source port. The backend must launch flows with exactly that source
  port, run them where the listener's kernel socket scan can see them, keep
  parallel slots isolated (`instance_id`), and write the
  `/tmp/iperf_{cport}_{i}.json` artifacts downstream parsers read.
- **Simulation** (RayNet): no kernel sockets — the backend exposes its own
  attach mechanism (RayNet: a flow-service RPC channel) and the key is purely
  a per-flow label.

## Reference implementations

- [`mininet/env.py`](mininet/env.py) — emulation: dumbbell topology, real
  iperf3 flows, live `tc` retuning, `wait()` sleeps the episode duration plus
  grace.
- [`raynet/env.py`](raynet/env.py) — simulation: OMNeT++ episode driven
  through a flow-service RPC in lockstep; the schedule and scenario ship in
  the episode config; `wait()` blocks on the simulator's completion signal.

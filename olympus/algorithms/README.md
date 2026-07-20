# Learner, Model, and Worker Architecture

Each Olympus algorithm lives in its own package:

```text
olympus/algorithms/<algorithm>/
├── __init__.py
├── learner.py
├── model.py
└── worker.py
```

The three main files exist because the algorithm definition, training process,
and live congestion-control process have different responsibilities and
runtime requirements.

`learner.py` and `worker.py` run as separate processes. `model.py` is imported
by both of them.

## Runtime Data Flow

```text
                         config.yaml
                              |
                              v
                      orchestrator.py
                       /            \
                      /              \
          starts one learner       starts episode slots
                  |                       |
                  |                 oc_bridge process
                  |                       |
                  |                 one worker per flow
                  |                       |
                  |<--- experiences ------|
                  |                       |
                  |---- new weights ----->|
                  |                       |
            saves checkpoints       reads TCP state
            and learner logs        chooses and applies CWND
```

The orchestrator selects the package from `runtime.algorithm`. The registry in
`olympus/common/registry.py` resolves the package's learner and worker scripts
without importing their heavy PyTorch modules into the orchestrator process.

## `model.py`: Algorithm Definition

`model.py` is normally the part adapted from an external implementation,
research repository, or paper. It describes what is being learned and how the
algorithm works mathematically. It normally owns:

- `STATE_DIM`, `ACTION_DIM`, and the action range or action-to-CWND mapping.
- State normalization, usually through
  `olympus/common/state_plugins.py`.
- The `Experience` or transition type sent over IPC.
- Actor, critic, world-model, or value-network definitions.
- Loss functions and algorithm-specific mathematical helpers.
- Checkpoint metadata and compatibility checks.
- Small inference helpers such as `actor.act(...)`.

When porting an algorithm, this is the closest thing to the "copy and paste"
file. Copy or adapt the external network definitions, losses, distributions,
and algorithm-specific math into `model.py`. Do not expect it to be a literal
unchanged copy: external observation shapes, action spaces, tensor layouts,
and checkpoint formats usually need to be converted to Olympus conventions.

Both the learner and worker import this module. The learner uses its networks
and losses for training, while the worker uses its policy code for live
actions. Keeping that code together prevents them from quietly using different
state layouts, network shapes, or action meanings. It also makes a serialized
`Experience` importable by both processes.

The model module should not own the training loop, manager server, socket
control loop, or experiment orchestration.

### State Compatibility

Most algorithms load the configured state plugin when `model.py` is imported:

```python
_STATE_PLUGIN = load_state_module("my_algorithm")
STATE_DIM = int(_STATE_PLUGIN.STATE_DIM)
normalize_state = _STATE_PLUGIN.normalize_state
```

The learner and worker receive the same `SAO_STATE` environment value, so they
must resolve the same state representation. Checkpoints and parameter
broadcasts should include `state_meta`, and both sides should reject
incompatible state dimensions or feature versions.

### Action Plugins

The action-to-TCP mapping is selected independently from the algorithm:

```yaml
runtime:
  action: cwnd_multiplier

actions:
  cwnd_multiplier:
    action_min: -1.0
    action_max: 1.0
    multiplier_min: 0.5
    multiplier_max: 2.0
    min_cwnd_change: 1

  astraea:
    action_min: -1.0
    action_max: 1.0
    step: 0.025
```

Action modules live in `olympus/actions/`. They define the policy action
dimension and range, convert policy output to a CWND multiplier, apply integer
CWND rounding, and expose compatibility metadata.

The built-in mappings are:

- `cwnd_multiplier`: the previous Olympus exponential mapping. A policy action
  of `-1`, `0`, or `1` means `0.5x`, `1x`, or `2x` CWND by default.
- `astraea`: the original Astraea relative update. Positive actions use
  `1 + step * action`; negative actions use
  `1 / (1 - step * action)`. Directional ceil/floor preserves a one-packet
  movement when a small relative update cannot be represented exactly.

`agent.cwnd_min` and `agent.cwnd_max` remain algorithm safety limits. The
selected action profile owns the policy and multiplier ranges, so those ranges
must not be duplicated in learner or agent configuration.

Learners and workers receive the same action selection through
`SAO_ACTION_NAME` and `SAO_ACTION_CONFIG`. Checkpoints record action metadata.
Changing action meaning requires a new run; a checkpoint trained with another
action mapping is rejected even when both mappings use `ACTION_DIM = 1`.

## `worker.py`: Live Rollout and Inference

A worker is attached to a live TCP flow by `oc_bridge`. It is latency
sensitive and should remain lightweight. Its responsibilities are:

1. Read the flow descriptors and runtime settings from environment variables.
2. Read TCP measurements with
   `tcp_sockopt.get_tcp_deepcc_info(OC_FLOW_FD)`.
3. Add any worker-maintained history needed by the state plugin.
4. Normalize the observation through `model.normalize_state(...)`.
5. Compute the reward through the selected reward plugin.
6. Run the execution policy and convert its action into a CWND update.
7. Apply the new CWND with `tcp_sockopt.set_cwnd(...)`.
8. Buffer and push experiences to the learner.
9. Periodically pull newer execution weights from the learner.
10. Write the per-flow trace used by plots and benchmarks.
11. Push a final `done=True` transition when the flow ends.

Important inputs include:

| Variable | Meaning |
| --- | --- |
| `OC_FLOW_FD` | File descriptor for the controlled TCP socket. |
| `OC_FLOW_ID` | Listener-assigned flow identifier. |
| `OC_CPORT` | Listener control port. |
| `SAO_CHECKPOINT` | Initial policy checkpoint. |
| `SAO_MANAGER_ADDR` / `SAO_MANAGER_KEY` | Learner IPC endpoint. |
| `SAO_REWARD` / `SAO_STATE` | Selected reward and observation plugins. |
| `SAO_INTERVAL_MS` | Control-loop period. |
| `SAO_CWND_MIN` / `SAO_CWND_MAX` | CWND safety bounds. |
| `SAO_DETERMINISTIC` | Disable rollout exploration for evaluation. |
| `SAO_REQUIRE_CHECKPOINT` | Fail instead of using fresh weights. |
| `SAO_TRACE_LOG` | Episode trace path. |

For inference, a worker must be able to run from a checkpoint without a
learner connection. Manager connection failures should therefore disable
experience upload and live weight refresh, not disable policy execution.

For full single-agent multi-flow support, the worker should also honor the
`SAO_LAGGED_POLICY_*` variables. They select background flows that use an older
checkpoint, deterministic actions, and usually no experience upload.

## `learner.py`: Optimization and Persistence

The learner is the central, long-lived training process. It normally:

1. Accepts `--config <resolved-config>` and `--port <learner-port>`.
2. Loads the selected `training` and `agent` configuration.
3. Creates networks, optimizers, replay or rollout storage, and target models.
4. Loads `training.resume_from` or the current rolling checkpoint.
5. Starts a `multiprocessing.managers.BaseManager` IPC server.
6. Receives experiences from every active worker.
7. Groups or samples those experiences according to the algorithm.
8. Runs gradient updates.
9. Publishes fresh execution parameters for workers.
10. Writes learner telemetry to `training.log_path`.
11. Saves the rolling model to `training.checkpoint`.
12. Saves on shutdown and closes the manager cleanly.

The orchestrator requires the learner to print this flushed readiness line:

```text
SAO_MANAGER_KEY=<hex-encoded-auth-key>
```

Without it, the orchestrator cannot authenticate workers or consider the
learner ready. The orchestrator also watches the learner and restarts it after
an unexpected exit.

Most current algorithms expose these manager methods:

```text
push_exp(exp)
push_exp_batch(exps)
pull_params()
```

Some algorithms add methods such as `push_bootstrap`, while MA-Dreamer exposes
the same operations through one `service()` proxy. The exact IPC shape is
algorithm-local, but `learner.py` and `worker.py` must agree exactly.

Only weights needed for execution should be broadcast. For example, TD3 sends
the actor but not the critics, while MA-Dreamer sends its local world model and
actor but keeps the global training model in the learner.

## Why Keep Three Components?

### Real-time isolation

The worker controls a TCP flow at a short interval. Replay sampling, backprop,
checkpoint writes, and GPU work in the learner must not stall that control
loop.

### One learner, many workers

Parallel episode slots and multi-flow episodes can all send data to one
learner. The learner combines experience and distributes one current policy
without workers sharing optimizer state.

### Training and inference use the same policy definition

Both processes import `model.py`, so checkpoint keys, network shapes, state
features, and action meanings stay aligned.

### Smaller execution surface

Workers need only inference-side networks. Training-only critics, target
networks, global world models, replay buffers, and optimizers remain in the
learner.

### Hot-swappable algorithms

The orchestrator only needs the package name and `MULTI_AGENT` flag. Algorithm
details stay inside the package instead of accumulating conditionals in the
orchestrator.

## Adding a New Algorithm

Adding a "new learner" means adding a complete algorithm package because the
learner, model, and worker form one versioned protocol.

Assume the new algorithm is named `my_algorithm`.

### What Should Be Copied?

The usual porting workflow is:

1. Start from the external algorithm's model and mathematical implementation.
   Adapt those parts into `model.py`.
2. Copy the closest Olympus `learner.py`, then replace its update logic with
   the new algorithm's training procedure.
3. Copy the closest Olympus `worker.py`, then replace its policy state, action
   call, and experience fields so they match the new model.

In short:

```text
External repository or paper
          |
          v
 model.py: networks, losses, actions, and algorithm math
          |
          +------> learner.py: trains those models
          |
          +------> worker.py: runs the policy on a TCP flow
```

Usually do not copy an external worker directly. Olympus workers contain
project-specific TCP socket access, CWND updates, learner IPC, reward plugins,
environment variables, trace logging, deterministic evaluation, and process
shutdown behavior. Those parts should come from the closest existing Olympus
worker.

### 1. Create the package

```text
olympus/algorithms/my_algorithm/
├── __init__.py
├── learner.py
├── model.py
└── worker.py
```

Declare how episodes should be launched:

```python
"""My algorithm."""

MULTI_AGENT = False
```

Use `MULTI_AGENT = True` only when the learner jointly trains all flows. A
single-agent algorithm still supports multi-flow environments through lagged
self-play and should remain `False`.

No central registry list needs editing. The registry imports
`olympus.algorithms.<runtime.algorithm>` by name.

### 2. Define `model.py` first

Copy or adapt the external model and algorithm math, then connect it to these
Olympus requirements:

- The state plugin and resulting state metadata.
- Action representation and CWND conversion.
- The exact `Experience` fields.
- Execution network APIs used by the worker.
- Training networks and losses used by the learner.
- Checkpoint compatibility and architecture metadata.

Use an existing package with the closest learning style as the starting point:

| New algorithm style | Closest examples |
| --- | --- |
| Off-policy actor-critic | `td3`, `orca`, `mbpo_td3` |
| Multi-agent off-policy CTDE | `ma_td3` |
| On-policy recurrent actor-critic | `recurrent_ppo` |
| Single-agent world model | `dreamer_v3` |
| Multi-agent world model | `ma_dreamer` |

### 3. Implement `worker.py`

Reuse the control-loop structure of the closest worker, then replace only the
algorithm-specific policy state, action selection, and experience fields.

The worker must keep these contracts aligned with `model.py`:

- Network constructor arguments and checkpoint keys.
- Recurrent state reset and carry behavior.
- Action stored in replay versus action applied to CWND.
- Observation shape and state metadata.
- Terminal transition semantics.
- Parameter broadcast payload keys.

Preserve deterministic checkpoint-only execution and the standard trace
columns so the existing benchmark and plotting tools continue to work.

### 4. Implement `learner.py`

Reuse the manager startup, signal handling, checkpoint paths, and telemetry
structure from the closest learner. Implement the new replay/rollout logic and
optimizer update inside that shell.

At minimum, verify:

- The manager methods match those registered by the worker.
- The readiness key is printed with `flush=True`.
- Fresh, rolling-checkpoint, and `resume_from` startup all work.
- Broadcast payloads contain everything the worker needs and nothing it does
  not need.
- Checkpoints contain execution weights, optimizer state where appropriate,
  `step`, architecture metadata, and state metadata.
- `SIGINT` and `SIGTERM` save and shut down cleanly.

### 5. Add configuration

Add an entry under `algorithms:` in `olympus/config.yaml`:

```yaml
algorithms:
  my_algorithm:
    training:
      resume_from: null
      save_every: 500
      param_broadcast_every: 50
      # Algorithm-specific optimizer and buffer settings.
    agent:
      interval_ms: 20
      cwnd_min: 10
      cwnd_max: 10000
      # Architecture and exploration settings.
```

Select it with:

```yaml
runtime:
  algorithm: my_algorithm
  reward: tempest
  state: kalman
```

The orchestrator copies the selected algorithm's `training`, `agent`, and
optional `reward` blocks into the resolved run config. It also rewrites
`training.checkpoint` and `training.log_path` into the new run directory.

### 6. Handle single-agent or multi-agent identity

For `MULTI_AGENT = False`, experiences are normally grouped by `traj_id` and
`step_in_traj`. In multi-flow environments, one live policy trains while
background flows may run lagged policy checkpoints.

For `MULTI_AGENT = True`, the orchestrator starts one listener and worker per
flow and supplies:

- `SAO_AGENT_ID`
- `SAO_N_AGENTS`
- `SAO_INSTANCE_ID`
- `SAO_CPORT_BASE`

Joint learners should include `agent_id`, a slot/episode `group_id`, and an
aligned `group_step` in each experience. This lets the learner reconstruct a
masked joint timestep despite workers reporting independently.

### 7. Add focused tests

At minimum, add tests that:

- Resolve the new worker and learner through
  `olympus.common.registry`.
- Assert the intended `MULTI_AGENT` value.
- Build the model and run one inference action.
- Serialize and deserialize one experience or parameter payload.
- Save and reload a checkpoint with state compatibility checks.
- Exercise one learner update on a small synthetic batch.
- For multi-agent learners, verify grouping, masking, and variable flow counts.

Useful commands are:

```bash
./venv_training/bin/python -m unittest olympus.tests.test_registry
./venv_training/bin/python -m unittest discover -s olympus/tests
```

## Common Failure Modes

- The learner and worker use different `Experience` field orders.
- The worker stores a pre-squash action while the critic expects a bounded
  post-squash action, or the reverse.
- The checkpoint omits state metadata and is loaded under a different state
  plugin.
- The worker constructor does not match the architecture saved by the learner.
- The learner prints logs but never prints `SAO_MANAGER_KEY=...`.
- The worker assumes a learner is always available, breaking inference.
- A true multi-agent learner is marked `MULTI_AGENT = False`, or a
  single-agent learner is marked `True`.
- A multi-agent worker omits `group_step`, preventing reliable cross-flow
  alignment.
- Training-only networks are broadcast to every worker, increasing IPC and
  inference overhead without changing actions.

The practical rule is simple: keep algorithm math in `model.py`, live
environment interaction in `worker.py`, and mutable training state in
`learner.py`.

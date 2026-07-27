# SAC continual-learning deployment

This service runs the recurrent Olympus SAC learner continuously against live
TCP flows.  Each listener-spawned flow process owns its recurrent actor and
fixed-cadence socket control loop.  It asynchronously uploads transitions to
one central learner and periodically pulls newer actor parameters.  Gradient
work, replay serialization, and checkpoint writes never run in the per-flow
control loop.

The service intentionally requires:

```yaml
runtime:
  algorithm: sac
  reward: proteus
  state: proteus
```

`proteus` uses only sender-observable goodput, RTT, observed minimum
RTT, and loss. It never reads the configured link capacity or base RTT.

## Run

Start from an SAC checkpoint whose state and action metadata match the config:

```bash
sudo -E ./venv_training/bin/python -m olympus.deployment_crl.run \
  --config olympus/deployment_crl/config.example.yaml \
  --checkpoint /absolute/path/to/sac_cwnd_model.pt
```

The first launch copies that model to
`deployment.continual.state_dir/sac_continual.pt`. Later launches resume the
rolling model and replay in that directory rather than overwriting them from
`--checkpoint`. To deliberately start a new learner, select a new empty state
directory.

The service writes:

```text
state_dir/
  config.resolved.yaml
  sac_continual.pt
  sac_replay.pkl
  learner_metrics.csv
```

Shutdown with SIGINT or SIGTERM. The listener is stopped first; the learner is
then given time to atomically save the model and replay.

## Retaining learned behavior

Proteus enables recurrent actor/critic distillation for continual deployment.
The seed checkpoint becomes a frozen teacher. A bounded reservoir keeps
sequence anchors sampled uniformly over the lifetime of the service, rather
than losing every old regime to FIFO eviction. On anchor sequences, the actor
matches the teacher's pre-tanh Gaussian with a KL loss and both critics match
the teacher's Q estimates. These losses are added to the ordinary SAC losses.

Teacher networks and their source step are stored in `sac_continual.pt`; the
anchor reservoir is stored alongside normal replay in `sac_replay.pkl`. With
`teacher_update_every: 0`, restarts retain the original teacher. Set a positive
update interval only when automatic teacher refresh is desired. The CSV and
learner output expose actor/critic distillation loss, anchor count, and teacher
step.

Distillation preserves behavior selected by the anchor reservoir; it does not
by itself prove that behavior is useful. Regime evaluation and candidate
promotion remain separate safeguards.

## Online updates and exploration

`updates_per_transition` bounds optimization by newly received experience.
For example, `0.25` permits one SAC update per four received transitions.
`max_update_burst` limits accumulated work after an idle or replay-warmup
period.

Exploration is disabled by default, so serving uses the SAC distribution mean
while still learning off-policy from observed transitions. Set
`deployment.continual.exploration: true` only for a controlled experiment; it
makes every attached flow sample the stochastic SAC policy. CWND bounds remain
active in both modes.

This is direct online policy updating, suitable for an instrumented research
deployment. It does not yet implement champion/candidate evaluation or gated
promotion. Use deterministic mode until a canary/promotion layer is added if
unreviewed live policy changes would be unsafe.

# Olympus per-flow inference deployment

This is the non-batched implementation. Every detected Astraea TCP socket gets
a fresh OS process. That process owns the duplicated socket FD, checkpoint
weights, model, recurrent state, state normalizer, control loop, and trace.
There is no inference server, Unix inference socket, shared model, or dynamic
batch queue.

Run only one deployment implementation at a time. Both launchers use the same
service lock and refuse to start if the other implementation is active.

## Algorithm support

Every algorithm works here, and nothing in this directory is algorithm-aware.
`olympus/algorithms/<name>/model.py` exposes

```python
build_policy(ckpt, agent_cfg=None, training_cfg=None,
             device='cpu', deterministic=True) -> Policy
```

where `Policy.act(state)` returns the same bounded action the algorithm's
worker feeds to `action_plugin.apply_cwnd`, and `Policy.reset()` drops carried
state (LSTM/GRU cells, RSSM latents, observation history). The contract and its
adapters live in `olympus/common/policy.py`; a new algorithm deploys as soon as
it ships a `build_policy`, with no edit to `deployment/`.

`olympus/tests/test_deployment_policies.py` rebuilds actors the way each
worker does and requires `build_policy` to produce identical actions, so the
deployed policy cannot silently drift from the trained one.

The base congestion control the listener attaches to comes from
`deployment.discovery.cc_algorithm`, else the config's `listener_cc`, else
`astraea`; override with `--cc-name`. For Orca, `agent.cubic_warmup` /
`cubic_warmup_s` (or the training config's `cubic_warmup_max_s`) keep the
kernel CC in charge for a fixed window — one second by default — after which
the agent takes over permanently. The handoff is purely time-based here;
`algorithms/orca/worker.py` additionally releases on the first loss.

The batched implementation still carries its own construction and covers only
`td3`, `mbpo_td3`, `dreamer_v3` and `ma_dreamer` — batching needs a batched
forward per algorithm, which `build_policy` (one flow, one call) does not
provide.

Run it with the checkpoint's resolved training config:

```bash
sudo -E ./venv_training/bin/python -m olympus.deployment.run \
  --config /path/to/config.resolved.yaml \
  --checkpoint /path/to/ma_dreamer_cwnd_model.pt
```

The separately invoked iperf traffic-generator command is unchanged:

```bash
./venv_training/bin/python -m olympus.deployment.public_iperf \
  --server SERVER --port 5201 --flows 4 --duration 40 --cc astraea \
  --output results/public_4flow
```

All `--flows` streams belong to one iperf3 control session and begin together.
Standard iperf3 does not support a different start delay for each `-P` stream.

For staggered independent flows, the server-pool command uses one remote server
per flow. Its defaults are `speedtest.ip-projects.de`,
`iperf3.phoenixremoteaccess.uk`, and `iperf.astra.in.ua`:

```bash
./venv_training/bin/python -m olympus.deployment.public_iperf_pool \
  --duration 40 --flow-start-delay 4 --cc astraea \
  --output results/server_pool
```

For the shared-model dynamic-batch implementation, use
`python -m olympus.deployment_batch.run` instead.

For recurrent SAC deployment that continuously collects transitions, trains
with the ground-truth-free `proteus` reward, persists replay, and
hot-refreshes live actor weights, use `python -m olympus.deployment_crl.run`;
see `olympus/deployment_crl/README.md` for its safety model and configuration.

# Olympus dynamic-batch inference deployment

This deployment keeps one OS process per TCP socket while loading the policy
once in a central dynamic-batching server. The existing `astraea_listener`
discovers sockets whose `TCP_CONGESTION` name matches `cc_algorithm`, duplicates
their FDs, and starts `worker.py` for each flow.

Build the listener, copy `config.example.yaml`, set an absolute checkpoint path,
then run as root (socket discovery and `pidfd_getfd` require it):

```bash
gcc -O2 -Wall -pthread -o astraea_listener astraea_listener.c
sudo -E ./venv_training/bin/python -m olympus.deployment_batch.run \
  --config olympus/deployment_batch/config.example.yaml
```

`td3`, `mbpo_td3`, `dreamer_v3`, and `ma_dreamer` actors are supported. For
MA-Dreamer, deployment uses the checkpoint's parameter-shared local world
model and actor; the global world model is training-only, so flows do not need
to be grouped or synchronized. Requests arriving within
`max_wait_us` are combined up to `max_batch_size`. Each flow's recurrent hidden
state is retained separately and removed when its worker disconnects.

Run only one deployment implementation at a time. Both launchers use the same
service lock and refuse to start if the per-flow implementation is active.

For a singleton-inference A/B run without changing the saved configuration,
override the batching limit on the service command:

```bash
sudo -E ./venv_training/bin/python -m olympus.deployment_batch.run \
  --config /path/to/config.resolved.yaml --checkpoint /path/to/model.pt \
  --max-batch-size 1
```

Per-flow trace CSVs append `s0` through `sN`, the normalized state vector sent
to the model for that inference request. This makes batched and singleton runs
directly comparable at the policy input boundary.

## Public iperf3 multi-flow benchmark

Start the long-running inference service in one terminal:

```bash
sudo -E ./venv_training/bin/python -m olympus.deployment_batch.run \
  --config olympus/config.yaml --checkpoint /path/to/model.pt
```

Run the independent traffic/plot command in another terminal:

```bash
sudo -E ./venv_training/bin/python -m olympus.deployment.public_iperf \
  --server SERVER --port 5201 \
  --flows 4 --duration 40 --cc astraea \
  --output results/public_4flow
```

The iperf command does not load a config or checkpoint and does not manage the
inference service. Both commands use `/tmp/olympus-deployment-traces` by
default; pass the same `--trace-dir` to each to choose another location. Its
`--output` directory is replaced on every run so it always contains one test
and one plot. Pass `--keep-history` to create timestamped runs instead.

Public iperf3 hosts and allowed ports change frequently, so the server is an
explicit argument. The output contains `ping.txt`, `iperf3.json`, extracted
`iperf3_server.json`, `selected_flows.json`, and `flows.png`. Per-flow CSVs
remain in the shared trace directory owned by the inference service.

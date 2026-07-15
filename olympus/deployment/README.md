# Olympus per-flow inference deployment

This is the non-batched implementation. Every detected Astraea TCP socket gets
a fresh OS process. That process owns the duplicated socket FD, checkpoint
weights, model, recurrent state, state normalizer, control loop, and trace.
There is no inference server, Unix inference socket, shared model, or dynamic
batch queue.

Run only one deployment implementation at a time. Both launchers use the same
service lock and refuse to start if the other implementation is active.

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

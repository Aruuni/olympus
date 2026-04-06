"""
training/inference_test.py — run one Mininet episode with the trained model
and plot CWND, RTT, throughput, and arm selection over time.

Usage:
  sudo -E env PATH="$PATH" HOME="$HOME" \\
    python training/inference_test.py \\
      --checkpoint training/oc_model_test.pt \\
      --listener   ./oc_listener \\
      --py         "$(pwd)/venv_training/bin/python" \\
      --bw 10 --delay 20 --duration 60 \\
      --output inference_plot.png
"""

import argparse
import os
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

from training.mininet_env import MininetEnv
from training.episode_plot import plot

# ── subprocess helpers ────────────────────────────────────────────────────────

def _terminate(proc):
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=8)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()

# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', default='training/oc_model_test.pt')
    ap.add_argument('--listener',   required=True, help='./oc_listener binary')
    ap.add_argument('--py',         required=True, help='Python binary for worker')
    ap.add_argument('--bw',         type=float, default=10.0,  help='link bw Mbps')
    ap.add_argument('--delay',      type=float, default=20.0,  help='one-way delay ms')
    ap.add_argument('--bdp-mult',   type=float, default=1.0)
    ap.add_argument('--loss',       type=float, default=None)
    ap.add_argument('--duration',   type=int,   default=60,    help='seconds')
    ap.add_argument('--cport',      type=int,   default=20000)
    ap.add_argument('--output',     default='inference_plot.png')
    args = ap.parse_args()

    checkpoint   = os.path.abspath(args.checkpoint)
    listener_bin = os.path.abspath(args.listener)
    python_bin   = os.path.abspath(args.py)
    worker_script = os.path.join(_HERE, 'oc_inference_worker.py')

    state_log  = '/tmp/oc_inference_state.csv'
    iperf_json = f'/tmp/iperf_{args.cport}_1.json'

    worker_env = dict(os.environ)
    worker_env.update({
        'OC_PYTHON':      python_bin,
        'OC_CHECKPOINT':  checkpoint,
        'OC_STATE_LOG':   state_log,
    })

    # Keep Astraea's background TF service wired up during inference tests.
    # If the caller did not provide explicit paths, default to the current
    # interpreter for the service and the repo-local script/config/model.
    worker_env.setdefault(
        'ASTRAEA_PYTHON',
        os.path.abspath(sys.executable),
    )
    worker_env.setdefault(
        'ASTRAEA_SCRIPT',
        os.path.abspath(os.path.join(_ROOT, 'astraea_service.py')),
    )
    worker_env.setdefault(
        'ASTRAEA_CONFIG',
        os.path.abspath(os.path.join(_ROOT, 'astraea', 'astraea.json')),
    )
    worker_env.setdefault(
        'ASTRAEA_MODEL',
        os.path.abspath(os.path.join(_ROOT, 'astraea', 'models', 'exported')),
    )

    # Clean stale files
    for p in (state_log, iperf_json):
        try:
            os.remove(p)
        except FileNotFoundError:
            pass

    print(f'[inference_test] bw={args.bw}Mbps  delay={args.delay}ms  '
          f'duration={args.duration}s', flush=True)

    # ── Start Mininet ─────────────────────────────────────────────────────────
    env = MininetEnv(
        n=1, bw=args.bw, delay=args.delay,
        bdp_mult=args.bdp_mult, loss=args.loss,
        duration=args.duration, cport=args.cport,
    )
    env.start()

    # ── Start oc_listener ─────────────────────────────────────────────────────
    listener_cmd = [
        listener_bin,
        '--cport',   str(args.cport),
        '--worker',  worker_script,
        '--mode',    'mininet',
        '--scan-ms', '100',
    ]
    listener_proc = subprocess.Popen(listener_cmd, env=worker_env)

    # ── Run iperf3 ────────────────────────────────────────────────────────────
    env.run_iperf()
    print(f'[inference_test] iperf3 running for {args.duration}s ...', flush=True)
    time.sleep(args.duration + 4)

    # ── Tear down ─────────────────────────────────────────────────────────────
    _terminate(listener_proc)
    env.stop()

    # ── Plot ──────────────────────────────────────────────────────────────────
    output = os.path.abspath(args.output)
    if os.path.exists(state_log):
        plot(state_log, iperf_json, output,
             bw=args.bw, delay=args.delay)
    else:
        print('[inference_test] state log not found — did the worker start?',
              flush=True)


if __name__ == '__main__':
    main()

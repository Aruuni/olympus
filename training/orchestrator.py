"""
training/orchestrator.py — sequential Olympus training loop

Runs one Mininet episode at a time:
  1. Build Mininet topology
  2. Start oc_listener (which spawns oc_worker for the flow)
  3. Run iperf3 for `duration` seconds
  4. Kill oc_listener, tear down Mininet
  5. Repeat, cycling through the environments list

The oc_worker handles both inference and training inline; no separate
learner process is needed.

Usage:
  sudo -E env PATH="$PATH" HOME="$HOME" \\
    python training/orchestrator.py \\
      --config  training/config.yaml \\
      --listener ./oc_listener \\
      --py      $(pwd)/venv_training/bin/python
"""

import argparse
import itertools
import json
import os
import random
import signal
import subprocess
import sys
import threading
import time

import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

from training.mininet_env import MininetEnv
from training.episode_plot import plot as _plot_episode
from training.plot_returns_watcher import generate_plot as _plot_returns

# ── episode pool ──────────────────────────────────────────────────────────────

def _build_pool(cfg: dict) -> tuple:
    """
    Return (pool, shuffle) where pool is a list of episode-config dicts.

    Two config formats are supported:

    1. experiments: (old format) — fixed list, cycled in order.

    2. sweep: — Cartesian product of bws × delays × link_schedules.
       All combinations are generated and shuffled each pass.

       sweep:
         bws:    [10, 50, 100]
         delays: [20, 40, 80]
         link_schedules:
           - []                                       # no change
           - [{t: 30, bw: 20, delay: 60}]             # one change
           - [{t: 30, bw: 5}, {t: 60, bw: 50}]       # two changes
         # any other key (bdp_mult, flows, duration …) is passed through
    """
    if 'sweep' in cfg:
        sw       = cfg['sweep']
        bws      = sw.get('bws',    [sw.get('bw',    10.0)])
        delays   = sw.get('delays', [sw.get('delay', 20.0)])
        schedules = sw.get('link_schedules', [[]])
        base     = {k: v for k, v in sw.items()
                    if k not in ('bws', 'delays', 'link_schedules', 'bw', 'delay')}
        pool = []
        for bw, delay, sched in itertools.product(bws, delays, schedules):
            ep = dict(base)
            ep['bw']    = float(bw)
            ep['delay'] = float(delay)

            # Resolve relative schedule fields against this combo's base values:
            #   bw_frac    → bw    = base_bw    * bw_frac
            #   delay_frac → delay = base_delay * delay_frac
            resolved = []
            for entry in sched:
                e = dict(entry)
                if 'bw_frac' in e:
                    e['bw'] = round(float(bw) * e.pop('bw_frac'), 3)
                if 'delay_frac' in e:
                    e['delay'] = round(float(delay) * e.pop('delay_frac'), 1)
                resolved.append(e)
            ep['link_schedule'] = resolved
            pool.append(ep)
        return pool, True     # shuffle = True

    envs = cfg.get('experiments', cfg.get('environments', [{}]))
    return list(envs), False  # shuffle = False (cycle in original order)


# ── helpers ───────────────────────────────────────────────────────────────────

def _run_link_schedule(env, schedule: list, episode_start: float, stop: threading.Event) -> None:
    """
    Background thread: fire tc changes at the times declared in the schedule.

    Each entry is a dict with keys:
      t    — seconds after episode_start to apply this change
      bw   — new bottleneck bandwidth in Mbps  (optional)
      delay— new one-way propagation delay in ms (optional)
      loss — new loss percentage (optional)

    The entries must be sorted by ascending t.
    """
    for entry in schedule:
        t_target = episode_start + float(entry['t'])
        while not stop.is_set():
            remaining = t_target - time.monotonic()
            if remaining <= 0:
                break
            stop.wait(timeout=min(remaining, 0.05))
        if stop.is_set():
            return
        bw    = entry.get('bw',    None)
        delay = entry.get('delay', None)
        loss  = entry.get('loss',  None)
        try:
            env.set_link(bw=bw, delay=delay, loss=loss)
            print(f'[orch] link change  t={time.monotonic()-episode_start:.1f}s'
                  f'  bw={bw}  delay={delay}  loss={loss}', flush=True)
        except Exception as e:
            print(f'[orch] link change failed: {e}', flush=True)


def _terminate(proc):
    """
    Kill proc and its entire process group (workers + astraea subprocesses).
    Works because oc_listener is spawned with start_new_session=True.

    We always follow SIGTERM with SIGKILL on the same saved pgid — astraea_service
    (TF) ignores SIGTERM, and oc_listener dies fast enough that a second
    os.getpgid() call after proc.wait() would raise ProcessLookupError.
    """
    if proc is None or proc.poll() is not None:
        return
    try:
        pgid = os.getpgid(proc.pid)   # save before any killing
    except ProcessLookupError:
        return
    # SIGTERM first (graceful), then unconditional SIGKILL on the same pgid
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(pgid, sig)
        except ProcessLookupError:
            break
        try:
            proc.wait(timeout=3)
            break
        except subprocess.TimeoutExpired:
            pass

# ── single episode ────────────────────────────────────────────────────────────

def run_episode(cfg: dict, ecfg: dict, episode: int,
                listener_bin: str, python_bin: str) -> None:

    t_cfg     = cfg.get('training', cfg)   # support flat or nested YAML
    cport     = cfg.get('cport_base', 20000)
    duration  = ecfg.get('duration', cfg.get('duration', 60))
    checkpoint = t_cfg.get('checkpoint', 'training/oc_model.pt')
    log_csv    = checkpoint.replace('.pt', '_log.csv')

    # Environment variables forwarded to the worker via oc_listener
    worker_env = dict(os.environ)
    worker_env.update({
        'OC_PYTHON':      python_bin,
        'OC_CHECKPOINT':  os.path.abspath(checkpoint),
        'OC_BATCH_SIZE':  str(t_cfg.get('batch_size',  32)),
        'OC_MIN_REPLAY':  str(t_cfg.get('min_replay',  64)),
        'OC_LR':          str(t_cfg.get('lr',           3e-4)),
        'OC_GAMMA':       str(t_cfg.get('gamma',        0.99)),
        'OC_SAVE_EVERY':  str(t_cfg.get('save_every',   20)),
        'OC_C_ENTROPY':   str(t_cfg.get('c_entropy',   0.01)),
        'OC_C_TERM':      str(t_cfg.get('c_term',      0.1)),
        'OC_C_DWELL':     str(t_cfg.get('c_dwell',     0.1)),
        'OC_LOG_CSV':       os.path.abspath(log_csv),
        'OC_DWELL_MIN_MS':  str(t_cfg.get('dwell_min_ms', 50)),
        'OC_DWELL_MAX_MS':  str(t_cfg.get('dwell_max_ms', 500)),
        'OC_ARMS':            ','.join(cfg.get('arms', ['cubic','bbr1','vegas','bbr3','astraea'])),
        'OC_REWARD_W_BW':   str(t_cfg.get('reward_w_bw',   0.5)),
        'OC_REWARD_W_RTT':  str(t_cfg.get('reward_w_rtt',  0.5)),
        'OC_REWARD_W_LOSS': str(t_cfg.get('reward_w_loss', 0.5)),
        # Episode-specific link parameters for reward normalisation
        'OC_LINK_BW':     str(float(ecfg.get('bw',    10.0))),
        'OC_LINK_DELAY':  str(float(ecfg.get('delay', 20.0))),   # RTT ms (not one-way)
        # Per-frame state log used to generate the episode plot
        'OC_STATE_LOG':   f'/tmp/oc_state_ep{episode}.csv',
    })

    # Optional Astraea background service — set if 'astraea' block present in config
    a_cfg = cfg.get('astraea', {})
    if a_cfg:
        worker_env.update({
            'ASTRAEA_PYTHON':  os.path.abspath(a_cfg.get('python',
                                   os.path.join(_ROOT, 'venv_astraea', 'bin', 'python'))),
            'ASTRAEA_SCRIPT':  os.path.abspath(a_cfg.get('script',
                                   os.path.join(_ROOT, 'astraea_service.py'))),
            'ASTRAEA_CONFIG':  os.path.abspath(a_cfg.get('config',
                                   os.path.join(_ROOT, 'astraea', 'astraea.json'))),
            'ASTRAEA_MODEL':   os.path.abspath(a_cfg.get('model',
                                   os.path.join(_ROOT, 'astraea', 'models', 'exported'))),
        })

    print(f'\n[orch] episode {episode}  '
          f'bw={ecfg.get("bw",10)}Mbps  delay={ecfg.get("delay",20)}ms  '
          f'duration={duration}s  cport={cport}', flush=True)

    # Schedule of link changes during this episode (list of {t, bw?, delay?, loss?})
    link_schedule = ecfg.get('link_schedule', [])

    # ── 1. Start Mininet ──────────────────────────────────────────────────────
    env = MininetEnv(
        n        = ecfg.get('flows', 1),
        bw       = ecfg.get('bw',   10.0),
        delay    = ecfg.get('delay',20.0),
        bdp_mult = ecfg.get('bdp_mult', 1.0),
        loss     = ecfg.get('loss',  None),
        duration = duration,
        cport    = cport,
    )
    env.start()

    # Stamp episode_start and encode the schedule before spawning the listener
    # so the worker process inherits both in its environment.
    episode_start = time.monotonic()
    worker_env['OC_EPISODE_START'] = str(episode_start)
    worker_env['OC_LINK_SCHEDULE'] = json.dumps(link_schedule)

    # ── 2. Start oc_listener ──────────────────────────────────────────────────
    worker_script = os.path.join(_HERE, 'oc_worker.py')
    listener_cmd = [
        listener_bin,
        '--cport',   str(cport),
        '--worker',  worker_script,
        '--mode',    'mininet',
        '--scan-ms', str(cfg.get('scan_ms', 100)),
    ]
    listener_proc = subprocess.Popen(listener_cmd, env=worker_env,
                                      start_new_session=True)

    # ── 3. Run iperf3 + link-change scheduler ────────────────────────────────
    env.run_iperf()
    print(f'[orch] iperf3 running for {duration}s ...', flush=True)

    sched_stop = threading.Event()
    if link_schedule:
        sched_thread = threading.Thread(
            target=_run_link_schedule,
            args=(env, link_schedule, episode_start, sched_stop),
            daemon=True,
        )
        sched_thread.start()
    else:
        sched_thread = None

    time.sleep(duration + 3)   # +3s for iperf to finish and listener to react

    # ── 4. Tear down ──────────────────────────────────────────────────────────
    print(f'[orch] episode {episode} done — stopping', flush=True)
    sched_stop.set()
    if sched_thread:
        sched_thread.join(timeout=2)
    _terminate(listener_proc)
    env.stop()
    time.sleep(2)   # let OVS finish tearing down bridges before next episode

    # ── 5. Episode plot ───────────────────────────────────────────────────────
    state_log  = worker_env['OC_STATE_LOG']
    iperf_json = f'/tmp/iperf_{cport}_1.json'
    plots_dir  = os.path.join(_HERE, 'plots')
    bw_str     = f"{ecfg.get('bw', 10):.0f}"
    delay_str  = f"{ecfg.get('delay', 20):.0f}"
    pdf_path   = os.path.join(plots_dir,
                              f'ep{episode:04d}_bw{bw_str}_d{delay_str}.pdf')
    os.makedirs(plots_dir, exist_ok=True)
    ep_return = _plot_episode(
        state_log_path  = state_log,
        iperf_json_path = iperf_json,
        output          = pdf_path,
        bw              = float(ecfg.get('bw',    10.0)),
        delay           = float(ecfg.get('delay', 20.0)),
        title           = (f'Episode {episode}  —  bw={bw_str} Mbps  '
                           f'delay={delay_str} ms  '
                           f'({"scheduled" if link_schedule else "static"})'),
        link_schedule   = link_schedule,
    )

    # Log episode return to a CSV for cross-episode reward tracking
    returns_csv = os.path.join(_HERE, 'plots', 'episode_returns.csv')
    write_header = not os.path.exists(returns_csv)
    with open(returns_csv, 'a', newline='') as f:
        import csv as _csv
        w = _csv.writer(f)
        if write_header:
            w.writerow(['episode', 'bw', 'delay', 'scheduled', 'return'])
        w.writerow([episode, ecfg.get('bw', 10), ecfg.get('delay', 20),
                    int(bool(link_schedule)),
                    f'{ep_return:.4f}' if ep_return is not None else ''])

    # Re-plot the full returns curve every 10 completed episodes
    if (episode + 1) % 10 == 0:
        _plot_returns()

# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--config',   required=True,  help='YAML config file')
    ap.add_argument('--listener', required=True,  help='path to oc_listener binary')
    ap.add_argument('--py',       required=True,  help='Python binary for oc_worker')
    ap.add_argument('--episodes', type=int, default=None,
                    help='number of episodes to run (default: run forever)')
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    listener_bin = os.path.abspath(args.listener)
    python_bin   = os.path.abspath(args.py)
    n_episodes   = args.episodes or cfg.get('episodes', None)

    pool, do_shuffle = _build_pool(cfg)
    print(f'[orch] pool size={len(pool)}  shuffle={do_shuffle}', flush=True)

    pending: list = []   # current pass — refilled (and shuffled) when empty
    episode = 0

    try:
        while n_episodes is None or episode < n_episodes:
            if not pending:
                pending = list(pool)
                if do_shuffle:
                    random.shuffle(pending)
                print(f'[orch] new pass over {len(pending)} episodes'
                      f'{"  (shuffled)" if do_shuffle else ""}', flush=True)
            ecfg = pending.pop(0)
            try:
                run_episode(cfg, ecfg, episode, listener_bin, python_bin)
            except Exception as e:
                print(f'[orch] episode {episode} error: {e}', flush=True)
            episode += 1
    except KeyboardInterrupt:
        print('\n[orch] interrupted', flush=True)

    print('[orch] done', flush=True)


if __name__ == '__main__':
    main()

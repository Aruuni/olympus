"""Per-flow rollout worker for Recurrent PPO.

On every 20 ms control tick: read DeepCC state → encode → sample action →
set_cwnd → push (s, a_raw, log_prob, value, r, done) to the learner. The
learner buffers full rollouts, runs K-epoch clipped-PPO updates, and
broadcasts new weights; workers pick them up on their next pull.

Same SAO_* / OC_* env-var contract as the other algorithms in this package.
"""

import csv
import io
import os
import sys
import threading
import time

import numpy as np

os.environ.setdefault('OMP_NUM_THREADS',      '1')
os.environ.setdefault('MKL_NUM_THREADS',      '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS',  '1')

import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

from multiprocessing.managers import BaseManager

_HERE = os.path.dirname(os.path.abspath(__file__))
_ALG_DIR = os.path.dirname(_HERE)
_PKG = os.path.dirname(_ALG_DIR)
_REPO = os.path.dirname(_PKG)
sys.path.insert(0, _REPO)

from olympus.algorithms.recurrent_ppo import model
from olympus.common import flow_backend
from olympus.common.action_plugins import load_action_module
from olympus.common.bbr_probe import BbrProbe
from olympus.common.registry import reward_module

_ACTION_PLUGIN = load_action_module()


class _Mgr(BaseManager):
    pass


_Mgr.register('push_exp')
_Mgr.register('push_exp_batch')
_Mgr.register('push_bootstrap')
_Mgr.register('pull_params')


_LOG_FLUSH_EVERY = 50


def _connect_manager():
    addr = os.environ.get('SAO_MANAGER_ADDR', '')
    key  = os.environ.get('SAO_MANAGER_KEY',  '')
    if not addr or not key:
        return None
    host, port = addr.rsplit(':', 1)
    mgr = _Mgr(address=(host, int(port)), authkey=bytes.fromhex(key))
    try:
        mgr.connect()
        return mgr
    except Exception as e:
        print(f'[worker] manager connect failed: {e}', flush=True)
        return None


def _srtt_ms(raw: dict) -> float:
    try:
        srtt_raw = float(raw.get('srtt_us', 0.0) or 0.0)
        fallback_us = float(raw.get('avg_urtt', 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0
    srtt_us = (srtt_raw / 8.0) if srtt_raw > 0.0 else fallback_us
    return max(srtt_us, 0.0) / 1e3


def _drain_state_fd(fd: int, stop: threading.Event):
    try:
        import select
        while not stop.is_set():
            r, _, _ = select.select([fd], [], [], 0.1)
            if r:
                os.read(fd, 4096)
    except Exception:
        pass


def run():
    reward_name   = os.environ.get('SAO_REWARD', 'tempest')
    reward_plugin = reward_module(reward_name)

    flow_fd  = int(os.environ['OC_FLOW_FD'])
    flow_id  = int(os.environ['OC_FLOW_ID'])
    cport    = int(os.environ.get('OC_CPORT', '0'))
    state_fd = int(os.environ.get('OC_STATE_FD', '-1'))
    episode  = int(os.environ.get('SAO_EPISODE', '0'))
    ckpt_path = os.environ.get('SAO_CHECKPOINT', '')

    interval_ms = float(os.environ.get('SAO_INTERVAL_MS', '20'))
    cwnd_min = int(os.environ.get('SAO_CWND_MIN', '10'))
    cwnd_max = int(os.environ.get('SAO_CWND_MAX', '10000'))
    hidden   = int(os.environ.get('SAO_HIDDEN',   '256'))
    require_checkpoint = os.environ.get('SAO_REQUIRE_CHECKPOINT', '0') == '1'

    deterministic = os.environ.get('SAO_DETERMINISTIC', '0') == '1'
    simulation_backend = flow_backend.is_simulation_backend()

    state_log_path = os.environ.get('SAO_TRACE_LOG', '')
    if state_log_path and os.environ.get('SAO_TRACE_LOG_SUFFIX_BY_FLOW', '0') == '1':
        root, ext = os.path.splitext(state_log_path)
        state_log_path = f'{root}_flow{flow_id}{ext or ".csv"}'

    net = model.RecurrentPPONet(model.STATE_DIM, hidden)
    step = 0
    if ckpt_path and os.path.exists(ckpt_path):
        try:
            ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            model.assert_checkpoint_state_compatible(ckpt, source=ckpt_path)
            net.load_state_dict(ckpt['model'])
            step = ckpt.get('step', 0)
            print(f'[worker cport={cport}] loaded checkpoint step={step}', flush=True)
        except Exception as e:
            print(f'[worker cport={cport}] ckpt load failed ({e}) - fresh init', flush=True)
            if require_checkpoint:
                raise
    elif require_checkpoint:
        raise FileNotFoundError(f'[worker cport={cport}] checkpoint not found: {ckpt_path}')
    net.eval()

    mgr = _connect_manager()

    stop_drain = threading.Event()
    if state_fd >= 0:
        threading.Thread(target=_drain_state_fd, args=(state_fd, stop_drain),
                         daemon=True).start()

    log_file, log_writer = None, None
    if state_log_path:
        os.makedirs(os.path.dirname(state_log_path), exist_ok=True)
        log_file = open(state_log_path, 'w', newline='')
        log_writer = csv.writer(log_file)
        log_writer.writerow([
            't_s', 'option', 'cwnd_mult', 'cwnd',
            'avg_thr_mbps', 'avg_urtt_ms', 'srtt_ms', 'min_rtt_ms',
            'loss_ratio', 'reward', 'kalman_rtt_ms',
            'act_mu', 'act_sig',
        ] + [f's{i}' for i in range(model.STATE_DIM)])

    reward_calc = reward_plugin.make_reward_calc()
    t0 = float(os.environ.get('SAO_EPISODE_START', '0')) or time.monotonic()
    traj_id = f'recurrent_ppo_{cport}_{episode}_{flow_id}'

    prev_state      = None
    prev_action_raw = 0.0
    prev_log_prob   = 0.0
    prev_value      = 0.0
    prev_urtt       = 0.0
    prev_cwnd       = 10
    peak_thr        = 0.0
    step_in_traj    = 0
    ever_alive      = False

    gru_h = torch.zeros(1, 1, net.hidden)
    interval_s = interval_ms / 1000.0

    weight_pull_counter = 0
    weight_pull_every = int(os.environ.get('SAO_WEIGHT_PULL_EVERY', '50'))

    push_every = max(1, int(os.environ.get('OC_PUSH_EVERY', '16')))
    exp_buf = []
    log_flush_counter = 0

    dead_flow_ms = float(os.environ.get('SAO_DEAD_FLOW_MS', '1000'))
    dead_steps = 0
    dead_steps_limit = int(dead_flow_ms / interval_ms)

    last_norm_s = None
    last_gru_h  = None

    # BBR3-style independent cwnd probe; reward- and algorithm-agnostic.
    probe = BbrProbe.from_env()

    print(f'[worker alg=recurrent_ppo reward={reward_name} cport={cport} flow={flow_id}] '
          f'started interval={interval_ms}ms det={int(deterministic)} hidden={hidden}',
          flush=True)

    try:
        while True:
            t_step_start = time.monotonic()

            try:
                raw = flow_backend.get_tcp_deepcc_info(flow_fd)
            except flow_backend.SimulationFinished:
                print('[worker] RayNet simulation finished - exiting',
                      flush=True)
                break
            except Exception as e:
                print(f'[worker] get_tcp_deepcc_info failed: {e} - exiting', flush=True)
                break

            avg_thr = float(raw.get('avg_thr', 0))
            peak_thr = max(peak_thr, avg_thr)

            if avg_thr <= 0:
                dead_steps += 1
            else:
                dead_steps = 0
                ever_alive = True
            flow_active = (not ever_alive) or (dead_steps < dead_steps_limit)

            raw['prev_urtt'] = prev_urtt
            raw['prev_cwnd'] = prev_cwnd
            raw['peak_thr']  = peak_thr
            raw['interval_ms'] = interval_ms
            norm_s = model.normalize_state(raw)
            reward = float(reward_calc.step(raw))
            urtt = float(raw.get('avg_urtt', 0))
            if urtt > 0:
                prev_urtt = urtt
            prev_cwnd = int(raw.get('cwnd', prev_cwnd))

            t_s = flow_backend.episode_seconds(raw, t0, wall_now=t_step_start)

            # Push the previous-step transition with the just-observed reward.
            if prev_state is not None and flow_active and mgr:
                exp_buf.append(model.Experience(
                    state=prev_state,
                    action_raw=prev_action_raw,
                    log_prob=prev_log_prob,
                    value=prev_value,
                    reward=reward,
                    done=False,
                    traj_id=traj_id,
                    step_in_traj=step_in_traj,
                ))
                step_in_traj += 1
                if len(exp_buf) >= push_every:
                    try:
                        mgr.push_exp_batch(list(exp_buf))
                    except Exception:
                        for e in exp_buf:
                            try:
                                mgr.push_exp(e)
                            except Exception:
                                pass
                    exp_buf.clear()

            # Sample action from current policy
            (action_raw, mult, log_prob, value, gru_h_new,
             act_mu, act_sig) = net.act(norm_s, gru_h, deterministic=deterministic)

            cur_cwnd = int(raw.get('cwnd', 10))
            desired_cwnd = _ACTION_PLUGIN.apply_cwnd(
                cur_cwnd, math.tanh(action_raw), cwnd_min, cwnd_max)

            srtt_raw_us = float(raw.get('srtt_us', 0) or 0)
            filter_rtt_us = (srtt_raw_us / 8.0) if srtt_raw_us > 0 else float(raw.get('avg_urtt', 0) or 0)
            clock_t = flow_backend.observation_clock(raw, wall_now=t_step_start)
            probe.observe_rtt(clock_t, filter_rtt_us)
            actual_cwnd, agent_locked, _ = probe.decide(
                clock_t, cur_cwnd, desired_cwnd)
            new_cwnd = int(np.clip(actual_cwnd, cwnd_min, cwnd_max))
            try:
                flow_backend.set_cwnd(flow_fd, new_cwnd)
            except Exception as e:
                print(f'[worker] set_cwnd failed: {e} - exiting', flush=True)
                break

            if agent_locked:
                # Probe owned this step; drop the transition so PPO doesn't
                # build a (s_pre, a_pre, r_probe, s_probe) tuple. The GRU
                # hidden state still advances to keep the recurrent context
                # consistent with what the actor saw.
                prev_state      = None
                last_norm_s     = None
                last_gru_h      = None
                gru_h           = gru_h_new
            else:
                prev_state      = norm_s
                prev_action_raw = action_raw
                prev_log_prob   = log_prob
                prev_value      = value
                last_norm_s     = norm_s
                last_gru_h      = gru_h
                gru_h           = gru_h_new

            weight_pull_counter += 1
            if mgr and weight_pull_counter >= weight_pull_every:
                weight_pull_counter = 0
                try:
                    param_bytes = mgr.pull_params()
                    if param_bytes is not None:
                        payload = torch.load(io.BytesIO(param_bytes),
                                             map_location='cpu', weights_only=False)
                        model.assert_checkpoint_state_compatible(
                            payload, source='learner payload')
                        net.load_state_dict(payload['state_dict'])
                        net.eval()
                        step = payload.get('step', step)
                except Exception:
                    pass

            if log_writer:
                log_writer.writerow([
                    f'{t_s:.3f}', 0, f'{mult:.3f}', new_cwnd,
                    f'{avg_thr * 8 / 1e6:.3f}',
                    f'{float(raw.get("avg_urtt", 0)) / 1e3:.3f}',
                    f'{_srtt_ms(raw):.3f}',
                    f'{float(raw.get("min_rtt", 0)) / 1e3:.3f}',
                    f'{float(raw.get("loss_ratio", 0)):.1f}',
                    f'{reward:.4f}',
                    f'{float(raw.get("kalman_min_rtt_us", 0.0)) / 1e3:.3f}',
                    f'{act_mu:.4f}', f'{act_sig:.4f}',
                ] + [f'{float(x):.5f}' for x in norm_s])
                log_flush_counter += 1
                if log_flush_counter % _LOG_FLUSH_EVERY == 0:
                    log_file.flush()

            elapsed = time.monotonic() - t_step_start
            sleep_t = interval_s - elapsed
            if sleep_t > 0 and not simulation_backend:
                time.sleep(sleep_t)

    finally:
        if mgr and exp_buf:
            try:
                mgr.push_exp_batch(list(exp_buf))
            except Exception:
                for e in exp_buf:
                    try:
                        mgr.push_exp(e)
                    except Exception:
                        pass
            exp_buf.clear()
        if prev_state is not None and mgr:
            try:
                mgr.push_exp(model.Experience(
                    state=prev_state,
                    action_raw=prev_action_raw,
                    log_prob=prev_log_prob,
                    value=prev_value,
                    reward=0.0,
                    done=True,
                    traj_id=traj_id,
                    step_in_traj=step_in_traj,
                ))
            except Exception:
                pass
        # Bootstrap value at the last observed state, used by the learner's GAE
        # for the truncated tail of this trajectory.
        if mgr and last_norm_s is not None:
            try:
                with torch.no_grad():
                    s_t = torch.from_numpy(last_norm_s).unsqueeze(0).unsqueeze(0)
                    _, _, v_t, _ = net.forward_sequence(s_t, last_gru_h)
                    bootstrap_value = float(v_t[0, -1].item())
                mgr.push_bootstrap(traj_id, bootstrap_value)
            except Exception:
                pass

        stop_drain.set()
        if log_file:
            try:
                log_file.flush()
            except Exception:
                pass
            log_file.close()
        print(f'[worker cport={cport} flow={flow_id}] done', flush=True)


if __name__ == '__main__':
    run()

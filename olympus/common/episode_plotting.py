"""Shared episode plotting helpers for socket-state traces.

The orchestrator, benchmarks, and future runners should use this module as the
single entry point for rendering per-episode plots from the state CSVs emitted
by Olympus workers. The plot implementations live under ``olympus.plots``;
this file owns the common decisions around filenames, per-flow/per-agent trace
discovery, required-log validation, and return extraction.
"""

import glob
import os
import re

from olympus.plots.episode_plot import (
    episode_return as _episode_return,
    plot as _plot_episode,
)
from olympus.plots.multi_flow_episode_plot import (
    episode_return as _multi_episode_return,
    plot as _plot_multi_episode,
)


def as_bool(value, default=False):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in ('1', 'true', 'yes', 'y', 'on')
    return default


def should_plot_episode(outputs, episode):
    if not as_bool(outputs.get('plot_episodes'), default=True):
        return False
    every_n = int(outputs.get('plot_every_n', 1) or 1)
    if every_n <= 1:
        return True
    return int(episode) % every_n == 0


def raw_state_base_path(episodes_dir, alg_name, episode):
    return os.path.join(
        episodes_dir, f'{alg_name}_raw_state_ep{int(episode):06d}.jsonl')


def plot_filename_env_part(env_type, env_name, default_env_type='mininet'):
    parts = [str(env_type or default_env_type).strip() or default_env_type]
    name = str(env_name or '').strip()
    if name and name.lower() != parts[0].lower():
        parts.append(name)
    label = _slug('_'.join(parts))
    return f'_{label}' if label else ''


def plot_env_title(env_type, env_name, default_env_type='mininet'):
    env_type = str(env_type or default_env_type).strip() or default_env_type
    env_name = str(env_name or 'config').strip() or 'config'
    return f'env_type={env_type}  env={env_name}'


def state_log_has_observations(path):
    try:
        if not path or not os.path.exists(path) or os.path.getsize(path) <= 0:
            return False
        with open(path, newline='') as f:
            # Header + at least one observation row.
            for line_no, _ in enumerate(f):
                if line_no >= 1:
                    return True
        return False
    except OSError:
        return False


def ensure_state_logs_saved(outputs, episode, state_logs, expected_hint):
    if not as_bool(outputs.get('require_state_logs'), default=False):
        return
    logs = [p for p in (state_logs or []) if p]
    missing = [p for p in logs if not state_log_has_observations(p)]
    if logs and not missing:
        return
    detail = ', '.join(missing) if missing else str(expected_hint)
    raise RuntimeError(
        f'episode {episode} did not save a non-empty state observation CSV '
        f'(expected {detail})')


def render_episode_plots(*, outputs, episode, alg_name, state_log, ecfg,
                         backend_type, env_name, link_schedule, n_flows,
                         trim_tail_s, mode='single', slot_id=None):
    """Render per-episode sockopt/state plots and return the episode return.

    ``mode`` is ``"single"`` for single-agent rollouts and ``"multi"`` for
    joint multi-agent rollouts. Single-agent rollouts may still produce
    per-agent or per-flow logs; this helper detects those conventions and uses
    the right plotter.
    """
    plots_dir = os.path.abspath(outputs['plots_dir'])
    os.makedirs(plots_dir, exist_ok=True)

    plot_episodes = should_plot_episode(outputs, episode)
    root, ext = os.path.splitext(state_log)
    ext = ext or '.csv'
    bw_str = f"{ecfg.get('bw', 100):.0f}"
    delay_str = f"{ecfg.get('delay', 20):.0f}"
    env_title = plot_env_title(backend_type, env_name)
    env_part = plot_filename_env_part(backend_type, env_name or 'config')

    if mode == 'multi':
        return _render_multi_agent_plots(
            outputs=outputs,
            episode=episode,
            state_log=state_log,
            root=root,
            ext=ext,
            ecfg=ecfg,
            plots_dir=plots_dir,
            link_schedule=link_schedule,
            n_flows=n_flows,
            trim_tail_s=trim_tail_s,
            plot_episodes=plot_episodes,
            bw_str=bw_str,
            delay_str=delay_str,
            env_title=env_title,
            env_part=env_part,
            slot_id=slot_id,
        )

    return _render_single_agent_plots(
        outputs=outputs,
        episode=episode,
        state_log=state_log,
        root=root,
        ext=ext,
        ecfg=ecfg,
        plots_dir=plots_dir,
        link_schedule=link_schedule,
        n_flows=n_flows,
        trim_tail_s=trim_tail_s,
        plot_episodes=plot_episodes,
        bw_str=bw_str,
        delay_str=delay_str,
        env_title=env_title,
        env_part=env_part,
        slot_id=slot_id,
    )


def _render_multi_agent_plots(*, outputs, episode, state_log, root, ext, ecfg,
                              plots_dir, link_schedule, n_flows, trim_tail_s,
                              plot_episodes, bw_str, delay_str, env_title,
                              env_part, slot_id):
    ep_return = _multi_episode_return(
        state_log, n_agents=n_flows, trim_tail_s=trim_tail_s)
    expected_state_logs = [
        f'{root}_a{agent_id}{ext}' for agent_id in range(int(n_flows))
    ]
    existing_state_logs = [
        path for path in expected_state_logs if os.path.exists(path)
    ]
    if not existing_state_logs and os.path.exists(state_log):
        existing_state_logs = [state_log]
    required_state_logs = (
        existing_state_logs if existing_state_logs == [state_log]
        else expected_state_logs
    )
    ensure_state_logs_saved(
        outputs, episode, required_state_logs,
        ', '.join(required_state_logs) if required_state_logs else state_log)

    try:
        if plot_episodes:
            pdf_path = os.path.join(
                plots_dir,
                f'ep{episode:06d}{env_part}_bw{bw_str}_d{delay_str}_n{n_flows}.pdf')
            plot_title = (f'Episode {episode}  flows={n_flows}  '
                          f'bw={bw_str}Mbps  delay={delay_str}ms  '
                          f'{env_title}  '
                          f'({"scheduled" if link_schedule else "static"})')
            plotted_return = _plot_multi_episode(
                state_log_path=state_log,
                output=pdf_path,
                bw=float(ecfg.get('bw', 100.0)),
                delay=float(ecfg.get('delay', 20.0)),
                title=plot_title,
                link_schedule=link_schedule,
                n_agents=n_flows,
                trim_tail_s=trim_tail_s,
            )
            if plotted_return is not None:
                ep_return = plotted_return
            print(f'[ep_plot] ep={episode} -> {pdf_path}  return={ep_return}',
                  flush=True)
    except Exception as e:
        prefix = _slot_prefix(slot_id)
        print(f'{prefix}ep={episode} multi-flow plot failed: {e}', flush=True)

    return ep_return


def _render_single_agent_plots(*, outputs, episode, state_log, root, ext, ecfg,
                               plots_dir, link_schedule, n_flows, trim_tail_s,
                               plot_episodes, bw_str, delay_str, env_title,
                               env_part, slot_id):
    ep_return = None

    # Per-agent traces: parameter-shared multi-agent models (and single-agent
    # models, which still use the per-agent trace writer) write one trace per
    # agent as ``<root>_aK.csv`` rather than to ``state_log`` directly.
    agent_logs = sorted(
        glob.glob(f'{root}_a*{ext}'),
        key=lambda p: (int(re.search(r'_a(\d+)', p).group(1))
                       if re.search(r'_a(\d+)', p) else 10 ** 9),
    )
    if agent_logs:
        ensure_state_logs_saved(outputs, episode, agent_logs, f'{root}_a*{ext}')
        n_agents = len(agent_logs)
        ep_return = _multi_episode_return(
            state_log, n_agents=n_agents, trim_tail_s=trim_tail_s)
        if plot_episodes:
            try:
                pdf_path = os.path.join(
                    plots_dir,
                    f'ep{episode:06d}{env_part}_bw{bw_str}_d{delay_str}_n{n_agents}.pdf')
                plot_title = (f'Episode {episode}  flows={n_agents}  '
                              f'bw={bw_str}Mbps  delay={delay_str}ms  '
                              f'{env_title}  '
                              f'({"scheduled" if link_schedule else "static"})')
                plotted_return = _plot_multi_episode(
                    state_log_path=state_log,
                    output=pdf_path,
                    bw=float(ecfg.get('bw', 100.0)),
                    delay=float(ecfg.get('delay', 20.0)),
                    title=plot_title,
                    link_schedule=link_schedule,
                    n_agents=n_agents,
                    trim_tail_s=trim_tail_s,
                )
                if plotted_return is not None:
                    ep_return = plotted_return
                print(f'[ep_plot] ep={episode} -> {pdf_path}  '
                      f'return={ep_return}', flush=True)
            except Exception as e:
                prefix = _slot_prefix(slot_id)
                print(f'{prefix}ep={episode} multi-flow plot failed: {e}',
                      flush=True)
        return ep_return

    state_logs = []
    if os.path.exists(state_log):
        state_logs.append(state_log)
    if ecfg.get('per_flow_state_logs'):
        flow_logs = sorted(
            glob.glob(f'{root}_flow*{ext}'),
            key=lambda p: int(re.search(r'_flow(\d+)', p).group(1))
            if re.search(r'_flow(\d+)', p) else 10 ** 9,
        )
        if flow_logs:
            state_logs = flow_logs

    required_state_logs = (
        [f'{root}_flow{flow_id}{ext}' for flow_id in range(n_flows)]
        if ecfg.get('per_flow_state_logs') else [state_log]
    )
    ensure_state_logs_saved(
        outputs, episode, required_state_logs,
        ', '.join(required_state_logs))

    if state_logs:
        primary_log = next(
            (p for p in state_logs if re.search(r'_flow0(?:\.|$)', p)),
            state_logs[0],
        )
        try:
            for log_path in state_logs:
                flow_match = re.search(r'_flow(\d+)', log_path)
                flow_suffix = f'_flow{flow_match.group(1)}' if flow_match else ''
                if plot_episodes:
                    pdf_path = os.path.join(
                        plots_dir,
                        f'ep{episode:06d}{env_part}_bw{bw_str}_d{delay_str}{flow_suffix}.pdf')
                    plot_title = (f'Episode {episode}{flow_suffix}  '
                                  f'bw={bw_str}Mbps  delay={delay_str}ms  '
                                  f'{env_title}  '
                                  f'({"scheduled" if link_schedule else "static"})')
                    ret = _plot_episode(
                        state_log_path=log_path,
                        output=pdf_path,
                        bw=float(ecfg.get('bw', 100.0)),
                        delay=float(ecfg.get('delay', 20.0)),
                        title=plot_title,
                        link_schedule=link_schedule,
                        trim_tail_s=trim_tail_s,
                    )
                    if log_path == primary_log:
                        ep_return = ret
                    print(f'[ep_plot] ep={episode}{flow_suffix} -> {pdf_path}  '
                          f'return={ret}', flush=True)
                elif log_path == primary_log:
                    ep_return = _episode_return(
                        log_path, trim_tail_s=trim_tail_s)
        except Exception as e:
            prefix = _slot_prefix(slot_id)
            print(f'{prefix}ep={episode} plot failed: {e}', flush=True)

    return ep_return


def _slot_prefix(slot_id):
    return '' if slot_id is None else f'[slot={slot_id}] '


def _slug(text: str) -> str:
    text = re.sub(r'[^A-Za-z0-9_.-]+', '_', str(text).strip())
    return text.strip('_') or 'learner'

"""Shared helpers used by every olympus benchmark.

Everything that used to live in benchmark_responsiveness.responsiveness and
was imported by other benchmark scripts now lives here, so no benchmark
needs to import from another benchmark. The reconstructed flow-plan /
binning helpers that the deleted benchmark_convergence.convergence used
to provide are at the bottom of this file (see _binned_series,
_kernel_flows, _write_flow_plan).

Functions and constants here are byte-for-byte the same as the originals
in responsiveness.py prior to the move. responsiveness.py now re-imports
them from this module.
"""

import argparse
import csv
import copy
import json
import math
import multiprocessing
import os
import random
import shutil
import subprocess
import sys
import threading
import time
import traceback

import numpy as np
import yaml

from olympus.orchestrator import (
    _activate_runtime_blocks,
    _as_bool,
    _resolve_repo_path,
    _slug,
)
from olympus.common.checkpoint_config import (
    apply_model_config_from_checkpoint,
)

def _load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _load_benchmark_config(path: str) -> dict:
    cfg = _load_yaml(path)
    approaches_config = cfg.get('approaches_config')
    if not approaches_config:
        return cfg

    shared_path = _resolve_from(os.path.dirname(os.path.abspath(path)),
                                str(approaches_config))
    shared = _load_yaml(shared_path)
    merged = copy.deepcopy(shared)
    merged.update(cfg)
    if 'approaches' not in cfg and 'approaches' in shared:
        merged['approaches'] = shared['approaches']
    merged['_approaches_config_path'] = shared_path
    return merged


def _resolve_from(base_dir: str, path: str) -> str:
    if os.path.isabs(path):
        return path
    direct = os.path.abspath(os.path.join(base_dir, path))
    if os.path.exists(direct):
        return direct
    return _resolve_repo_path(path)


def _load_approach_runtime_config(
        approach: dict, benchmark_config_path: str) -> tuple:
    """Load the resolved training config owned by a learned approach."""
    kind = str(approach.get('kind', 'model')).lower()
    if kind in ('kernel', 'astraea', 'orca'):
        return {}, ''

    label = approach.get('_label') or _approach_label(approach)
    config_value = str(approach.get('config', '')).strip()
    if not config_value:
        raise SystemExit(
            f'[bench] model approach {label!r} must define config')
    config_path = _resolve_from(
        os.path.dirname(os.path.abspath(benchmark_config_path)),
        config_value)
    if not os.path.isfile(config_path):
        raise SystemExit(
            f'[bench] approach config not found for {label}: '
            f'{config_path}')
    approach['_config_path'] = config_path
    return _load_yaml(config_path), config_path


def _approach_label(approach: dict) -> str:
    """Benchmark data folder name for this approach.

    `data_folder` is the explicit storage identity under output_root. `name` is
    kept as the older alias so existing configs keep working.
    """
    if approach.get('data_folder'):
        return _slug(approach['data_folder'])
    if approach.get('data_dir'):
        return _slug(approach['data_dir'])
    if approach.get('name'):
        return _slug(approach['name'])
    return _slug(
        f"{approach.get('algorithm', 'algorithm')}_"
        f"{approach.get('state', 'state')}_"
        f"{approach.get('reward', 'reward')}_"
        f"{approach.get('environment', 'environment')}"
    )


SAGE_REWARD_LABELS = {
    'sage': 'SAGE',
    'sage_drift': 'SAGE + K-Drift',
    'sage_sparse': 'SAGE + Sparse',
    'sage_closeness': 'SAGE + Closeness',
    'sage_drift_sparse': 'SAGE + K-Drift + Sparse',
}


def _default_approach_plot_label(approach: dict) -> str:
    reward = str(approach.get('reward') or '').strip().lower()
    sage_label = SAGE_REWARD_LABELS.get(reward)
    if not sage_label:
        return _approach_label(approach)

    parts = []
    algorithm = str(approach.get('algorithm') or '').strip()
    state = str(approach.get('state') or '').strip()
    environment = str(approach.get('environment') or '').strip()
    if algorithm:
        parts.append(algorithm.replace('_', '-').upper())
    if state:
        parts.append(f'{state.title()} State')
    parts.append(sage_label)
    if environment:
        parts.append(environment)
    return ' / '.join(parts)


def _approach_plot_label(approach: dict) -> str:
    return str(approach.get('plot_label') or _default_approach_plot_label(approach))


def _approach_selection_keys(approach: dict) -> set:
    keys = {_approach_label(approach)}
    if approach.get('name'):
        keys.add(_slug(approach['name']))
    return {key for key in keys if key}


def _specific_plot_keys(cfg: dict) -> set:
    """Slugged data-folder keys naming the curated `specific_plots` subset.

    The shared benchmarks config may carry a `specific_plots` list that names
    the approaches each benchmark draws into its `*_specific` comparison plot
    (the kernel baselines, Astraea, and the 0.995-gamma credit MA-Dreamer).
    Entries are data_folder/name strings (or mappings carrying one). Returns an
    empty set when the config omits the list, in which case callers skip the
    extra plot.
    """
    keys = set()
    for entry in cfg.get('specific_plots') or []:
        if isinstance(entry, dict):
            entry = entry.get('data_folder') or entry.get('name') or ''
        text = str(entry).strip()
        if text:
            keys.add(_slug(text))
    return keys


def _row_in_specific(row: dict, keys: set) -> bool:
    """True when ``row``'s approach/data_folder is in the ``specific_plots`` set."""
    if not keys:
        return False
    for field in ('data_folder', 'approach', 'name'):
        value = row.get(field)
        if value and _slug(str(value)) in keys:
            return True
    return False


def _configured_approaches(cfg: dict, only: set = None) -> list:
    approaches = []
    for raw in cfg.get('approaches') or []:
        approach = dict(raw or {})
        label = _approach_label(approach)
        approach['_label'] = label
        selection_keys = _approach_selection_keys(approach)
        if only and not (selection_keys & only):
            continue
        approaches.append(approach)
    return approaches


def _validate_unique_data_folders(approaches: list) -> None:
    seen = {}
    duplicates = []
    for approach in approaches:
        label = approach['_label']
        if label in seen:
            duplicates.append(label)
        seen[label] = approach
    if duplicates:
        raise SystemExit(
            '[resp_bench] duplicate benchmark data_folder/name values selected: '
            + ', '.join(sorted(set(duplicates)))
            + '; give each entry a unique data_folder to avoid overwriting runs'
        )


def _approach_label_map(cfg: dict) -> dict:
    """Map approach data folder names to current YAML plot labels.

    Duplicate data folders are intentionally ignored here. They happen when two
    entries differ only by checkpoint/plot_label but point at the same storage
    folder; in that case the existing per-folder run_meta/metrics label is the
    only safe identity.
    """
    candidates = {}
    counts = {}
    for raw in cfg.get('approaches') or []:
        approach = dict(raw or {})
        label = _approach_label(approach)
        counts[label] = counts.get(label, 0) + 1
        candidates[label] = _approach_plot_label(approach)
    return {
        label: plot_label
        for label, plot_label in candidates.items()
        if counts.get(label, 0) == 1
    }


def _validate_approach(approach: dict) -> None:
    kind = str(approach.get('kind', 'model')).lower()
    if kind == 'kernel':
        required = ['kernel_cc']
    elif kind == 'astraea':
        # The kernel TCP CC is fixed at "astraea". The listener daemon and
        # exported model paths are sourced from the approach (with sensible
        # defaults), so no per-approach checkpoint is required.
        required = []
    elif kind == 'orca':
        # The original Orca C++/TF1 implementation is launched from its own
        # install dir (approach.orca.install_dir, defaulted); it ships its
        # own trained actor, so no per-approach checkpoint is required.
        required = []
    else:
        required = [
            'algorithm', 'state', 'reward', 'environment', 'checkpoint',
        ]
    missing = [key for key in required if not approach.get(key)]
    if missing:
        raise SystemExit(
            f"[resp_bench] approach {_approach_label(approach)!r} is missing: "
            + ', '.join(missing)
        )


def _available_kernel_ccs() -> set:
    try:
        with open('/proc/sys/net/ipv4/tcp_available_congestion_control') as f:
            return set(f.read().split())
    except OSError:
        return set()


def _ensure_kernel_cc_available(approach: dict) -> None:
    """Load and validate the TCP CC selected by a kernel approach."""
    if str(approach.get('kind', 'model')).lower() != 'kernel':
        return

    cc_name = str(approach.get('kernel_cc', '')).strip()
    if not cc_name:
        _validate_approach(approach)
    available = _available_kernel_ccs()
    if cc_name in available:
        return

    load_output = ''
    if hasattr(os, 'geteuid') and os.geteuid() == 0:
        module = str(approach.get('kernel_module') or f'tcp_{cc_name}')
        try:
            result = subprocess.run(
                ['modprobe', module],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
            load_output = result.stdout.strip()
        except OSError as exc:
            load_output = str(exc)
        available = _available_kernel_ccs()
        if cc_name in available:
            print(
                f'[bench] loaded kernel CC {cc_name!r} via {module!r}',
                flush=True,
            )
            return

    choices = ' '.join(sorted(available)) or '(unreadable)'
    detail = f'; modprobe output: {load_output}' if load_output else ''
    raise SystemExit(
        f'[bench] kernel CC {cc_name!r} is unavailable '
        f'(available: {choices}). Load its module before benchmarking'
        f'{detail}'
    )


def _safe_overwrite_dir(path: str, root: str) -> None:
    path_abs = os.path.abspath(path)
    root_abs = os.path.abspath(root)
    if path_abs == root_abs or not path_abs.startswith(root_abs + os.sep):
        raise SystemExit(f'[resp_bench] refusing to overwrite outside output_root: {path_abs}')
    if os.path.exists(path_abs):
        shutil.rmtree(path_abs)
    os.makedirs(path_abs, exist_ok=True)


def _restore_sudo_user_ownership(path: str) -> None:
    """Give sudo-created benchmark outputs back to the invoking user."""
    uid_raw = os.environ.get('SUDO_UID')
    gid_raw = os.environ.get('SUDO_GID')
    if not uid_raw or not gid_raw or not path or not os.path.exists(path):
        return
    try:
        uid = int(uid_raw)
        gid = int(gid_raw)
    except ValueError:
        return

    for root, dirs, files in os.walk(path):
        for name in dirs + files:
            item = os.path.join(root, name)
            try:
                os.chown(item, uid, gid)
            except OSError:
                pass
    try:
        os.chown(path, uid, gid)
    except OSError:
        pass


def _write_schedule(run_dir: str, schedule: dict, bench: dict) -> tuple:
    csv_path = os.path.join(run_dir, 'schedule.csv')
    json_path = os.path.join(run_dir, 'schedule.json')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['t', 'bw', 'delay'])
        writer.writeheader()
        writer.writerows(schedule['rows'])
    with open(json_path, 'w') as f:
        json.dump({
            'seed': schedule['seed'],
            'duration_s': int(bench['duration_s']),
            'change_interval_s': int(bench['change_interval_s']),
            'rows': schedule['rows'],
        }, f, indent=2)
    return csv_path, json_path


def _finite_float(value, default=math.nan) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def _write_measurement_csv(csv_path: str, rows: list,
                           measure_interval_s: float = 1.0) -> None:
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    interval = max(float(measure_interval_s or 1.0), 1e-6)
    buckets = {}
    for t_s, bandwidth_mbps in rows:
        t_s = _finite_float(t_s)
        bandwidth_mbps = _finite_float(bandwidth_mbps)
        if not (math.isfinite(t_s) and math.isfinite(bandwidth_mbps)):
            continue
        bucket = int(max(t_s - 1e-9, 0.0) // interval)
        buckets.setdefault(bucket, []).append(bandwidth_mbps)

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['time', 'bandwidth'])
        writer.writeheader()
        for bucket in sorted(buckets):
            vals = buckets[bucket]
            writer.writerow({
                'time': f'{(bucket + 1) * interval:.3f}',
                'bandwidth': f'{(sum(vals) / max(1, len(vals))):.6f}',
            })


def _parse_iperf_json(json_path: str, csv_path: str = None,
                      measure_interval_s: float = 1.0) -> dict:
    try:
        with open(json_path) as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {'goodput_mbps': math.nan, 'samples': [], 'rtt_samples_ms': []}

    samples = []
    rtt_samples_ms = []
    for interval in data.get('intervals', []) or []:
        total = interval.get('sum') or {}
        t_end = _finite_float(total.get('end'))
        bps = _finite_float(total.get('bits_per_second'))
        if math.isfinite(t_end) and math.isfinite(bps):
            samples.append((t_end, bps / 1e6))
        for stream in interval.get('streams') or []:
            rtt_us = _finite_float(stream.get('rtt'))
            stream_end = _finite_float(stream.get('end'), default=t_end)
            if math.isfinite(stream_end) and math.isfinite(rtt_us) and rtt_us > 0:
                # iperf3 reports TCP RTT in microseconds in the client JSON.
                rtt_samples_ms.append((stream_end, rtt_us / 1000.0))

    if csv_path:
        _write_measurement_csv(csv_path, samples, measure_interval_s)

    goodput_mbps = math.nan
    end = data.get('end') or {}
    zero_summary = math.nan
    for key in ('sum_received', 'sum', 'sum_sent'):
        bps = _finite_float((end.get(key) or {}).get('bits_per_second'))
        if math.isfinite(bps) and bps > 0.0:
            goodput_mbps = bps / 1e6
            break
        if math.isfinite(bps) and bps == 0.0:
            zero_summary = 0.0
    if not math.isfinite(goodput_mbps) and samples:
        goodput_mbps = sum(v for _, v in samples) / len(samples)
    if not math.isfinite(goodput_mbps) and math.isfinite(zero_summary):
        goodput_mbps = zero_summary
    return {
        'goodput_mbps': goodput_mbps,
        'samples': samples,
        'rtt_samples_ms': rtt_samples_ms,
    }


def _prepare_runtime_cfg(base_cfg: dict, approach: dict, checkpoint: str,
                         approach_dir: str, run_dir: str, plot_episodes: bool) -> dict:
    cfg = copy.deepcopy(base_cfg)
    approach_algorithm = str(approach['algorithm'])
    approach_state = str(approach['state'])
    approach_reward = str(approach['reward'])
    runtime = cfg.setdefault('runtime', {})
    runtime['algorithm'] = approach_algorithm
    runtime['state'] = approach_state
    runtime['reward'] = approach_reward
    cfg['environment'] = {'name': str(approach['environment'])}
    if checkpoint:
        apply_model_config_from_checkpoint(cfg, checkpoint)
        runtime = cfg.setdefault('runtime', {})
        runtime['algorithm'] = approach_algorithm
        runtime['state'] = approach_state
        runtime['reward'] = approach_reward
    _activate_runtime_blocks(cfg)

    cfg.setdefault('training', {})['checkpoint'] = checkpoint
    cfg.setdefault('outputs', {}).update({
        'run_dir': approach_dir,
        'checkpoints_dir': os.path.join(approach_dir, 'checkpoints'),
        'episodes_dir': run_dir,
        'plots_dir': run_dir,
        'telemetry_dir': os.path.join(approach_dir, 'telemetry'),
        'traces_dir': run_dir,
        'plot_episodes': bool(plot_episodes),
        'resolved_config': os.path.join(run_dir, 'config.resolved.yaml'),
    })
    cfg['training']['log_path'] = os.path.join(approach_dir, 'telemetry', 'benchmark_metrics.csv')
    cfg.setdefault('reward', {})
    return cfg


def _deterministic_env(cfg: dict) -> None:
    os.environ['SAO_DETERMINISTIC'] = '1'
    os.environ['OC_DETERMINISTIC'] = '1'
    os.environ['SAO_NOISE_STD'] = '0.0'
    os.environ['SAO_REQUIRE_CHECKPOINT'] = '1'
    os.environ['OC_ORACLE_RTT'] = '0'
    agent_cfg = cfg.setdefault('agent', {})
    if 'term_threshold_inf' in agent_cfg:
        agent_cfg['term_threshold'] = agent_cfg['term_threshold_inf']
    if 'epsilon_test' in agent_cfg:
        agent_cfg['epsilon_start'] = agent_cfg['epsilon_test']
        agent_cfg['epsilon_end'] = agent_cfg['epsilon_test']


def _append_csv(path: str, fields: list, row: dict) -> None:
    exists = os.path.exists(path)
    with open(path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, '') for key in fields})


def _read_metrics(path: str) -> list:
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, newline='') as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def _resume_rows_by_key(metrics_csv: str, fields: list, key_fn) -> dict:
    """Load an existing metrics.csv into ``{trial_key: newest_row}`` for resume.

    ``key_fn(row)`` returns the trial key for a row, or ``None`` to drop the row
    (blank, foreign, or unparsable rows). When several rows share a key the last
    one wins, so a re-run that overwrites an earlier errored attempt is honoured.
    Missing fields are backfilled to '' so the row can be re-emitted cleanly.
    Returns an empty dict when the file is absent or unreadable, which lets
    callers fall back to a fresh run.
    """
    rows = {}
    if not os.path.isfile(metrics_csv):
        return rows
    try:
        with open(metrics_csv, newline='') as f:
            for row in csv.DictReader(f):
                key = key_fn(row)
                if key is None:
                    continue
                for field in fields:
                    row.setdefault(field, '')
                rows[key] = row
    except OSError:
        return {}
    return rows


def _collect_approach_rows_with_labels(output_root: str, label_map: dict = None) -> list:
    rows = []
    label_map = label_map or {}
    for name in sorted(os.listdir(output_root)):
        path = os.path.join(output_root, name)
        metrics_csv = os.path.join(path, 'metrics.csv')
        if not os.path.isdir(path) or not os.path.exists(metrics_csv):
            continue
        meta_label = None
        try:
            with open(os.path.join(path, 'run_meta.json')) as f:
                meta_label = ((json.load(f) or {}).get('approach') or {}).get('plot_label')
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            pass
        for row in _read_metrics(metrics_csv):
            row = dict(row)
            # The output folder is the benchmark run identity. This keeps
            # reruns such as "<approach>_legacy" separate even if the per-run
            # metrics were produced from the same original approach label.
            row['approach'] = name
            row['data_folder'] = name
            row['plot_label'] = (
                label_map.get(name)
                or meta_label
                or row.get('plot_label')
                or name
            )
            rows.append(row)
    return rows


def _safe_unlink(path: str) -> None:
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass
    except OSError:
        pass


def _copy_if_exists(src: str, dst: str) -> bool:
    if not os.path.exists(src):
        return False
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)
    return True


_ASTRAEA_DEFAULTS = {
    'binary': 'astraea_listener',
    'script': 'astraea_service.py',
    'config': 'astraea/astraea.json',
    'model': 'astraea/models/py',
    'python_env': 'ASTRAEA_PYTHON',
    # Candidate locations for the astraea-compatible interpreter, in order.
    # The first one that exists wins. The astraea agent uses TF1-style
    # `tf.layers.dense`, so the system Python (Keras 3) is unusable.
    'python_fallbacks': [
        'astraea/venv_astraea/bin/python',
        'venv_astraea/bin/python',
    ],
    'cc_name': 'astraea',
    'idle_exit_ms': 500,
    'scan_ms': 10,
}


def _astraea_settings(approach: dict, bench: dict) -> dict:
    """Merge per-approach overrides on top of the repo defaults.

    Each key may be set under approach['astraea'] (preferred) or directly on
    the approach itself for convenience. Paths resolve relative to the repo
    root via _resolve_repo_path so the same config works regardless of cwd.
    """
    cfg = dict(_ASTRAEA_DEFAULTS)
    cfg['idle_exit_ms'] = int(bench.get('astraea_idle_exit_ms', cfg['idle_exit_ms']))
    cfg['scan_ms'] = int(bench.get('astraea_scan_ms', cfg['scan_ms']))
    overrides = dict(approach.get('astraea') or {})
    # Single-string override for backwards-compat; collapse to the list form.
    if 'python_fallback' in overrides:
        overrides['python_fallbacks'] = [overrides.pop('python_fallback')]
    for key, value in overrides.items():
        if value is not None:
            cfg[key] = value
    return cfg


def _resolve_astraea_python(settings: dict) -> str:
    """Pick the first interpreter that actually exists.

    Honors $ASTRAEA_PYTHON first; otherwise walks the configured fallbacks.
    Returns '' if nothing is found, in which case the caller errors out
    instead of letting the C listener fall back to /usr/bin/python3 (which
    has Keras 3 and breaks the TF1-style astraea agent).
    """
    explicit = os.environ.get(str(settings.get('python_env', 'ASTRAEA_PYTHON')), '')
    if explicit and os.path.exists(explicit):
        return explicit
    fallbacks = settings.get('python_fallbacks') or []
    if isinstance(fallbacks, str):
        fallbacks = [fallbacks]
    for candidate in fallbacks:
        resolved = _resolve_repo_path(str(candidate))
        if resolved and os.path.exists(resolved):
            return resolved
    return ''


def _start_astraea_listener(approach: dict, bench: dict, log_dir: str):
    """Spawn the astraea_listener daemon used by `kind: astraea` approaches.

    Returns the Popen handle (or None if disabled). The caller must invoke
    _stop_astraea_listener at teardown.
    """
    import subprocess

    settings = _astraea_settings(approach, bench)
    binary = _resolve_repo_path(str(settings['binary']))
    script = _resolve_repo_path(str(settings['script']))
    py_config = _resolve_repo_path(str(settings['config']))
    py_model = _resolve_repo_path(str(settings['model']))

    if not os.path.exists(binary):
        raise SystemExit(
            f'[resp_bench] astraea_listener binary missing at {binary}; '
            f'build it with `gcc -O2 -Wall -o astraea_listener astraea_listener.c`'
        )

    env = dict(os.environ)
    py_bin = _resolve_astraea_python(settings)
    if not py_bin:
        raise SystemExit(
            '[resp_bench] no astraea-compatible Python found. Set '
            'ASTRAEA_PYTHON to a venv with TF1-style support, or place one at '
            + ', '.join(settings.get('python_fallbacks') or [])
        )
    env['ASTRAEA_PYTHON'] = py_bin

    cmd = [
        binary,
        '--mode', 'mininet',
        '--cc-name', str(settings['cc_name']),
        '--script', script,
        '--config', py_config,
        '--model', py_model,
        '--scan-ms', str(int(settings['scan_ms'])),
        '--switcher', '0',
        '--idle-exit-ms', str(int(settings['idle_exit_ms'])),
    ]

    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, 'astraea_listener.log')
    log_fh = open(log_path, 'a', buffering=1)
    log_fh.write(f'\n=== astraea_listener spawn {time.strftime("%Y-%m-%d %H:%M:%S")} ===\n')
    log_fh.write(' '.join(cmd) + '\n')
    log_fh.flush()
    proc = subprocess.Popen(
        cmd, stdout=log_fh, stderr=subprocess.STDOUT, env=env, close_fds=True,
    )
    print(f'[resp_bench] astraea_listener pid={proc.pid} python={py_bin} log={log_path}',
          flush=True)
    # Give the daemon a moment so the first ss scan completes before flows start.
    time.sleep(1.0)
    return proc, log_fh


def _stop_astraea_listener(handle) -> None:
    if not handle:
        return
    proc, log_fh = handle
    if proc and proc.poll() is None:
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except Exception:
            try:
                proc.kill()
                proc.wait(timeout=2)
            except Exception:
                pass
    if log_fh:
        try:
            log_fh.close()
        except Exception:
            pass


_ORCA_DEFAULTS = {
    # Where the original Orca repo is installed (must contain sender.sh,
    # receiver.sh and rl-module/{orca-server-mahimahi,clientThr}).
    'install_dir': os.environ.get('ORCA_INSTALL_DIR', '~/mininettestbed/CC/Orca'),
    # Base TCP port; each (slot, flow) gets port_base + slot*100 + flow.
    'port_base': 47000,
    # Extra wall-clock seconds added on top of the episode duration so the
    # TF1 actor has time to load its checkpoint before the duration timer
    # starts, plus teardown. Orca's actor warmup is heavy (~10-40 s).
    'warmup_slack_s': 60.0,
    # Start the server this long before its clientThr so the listening
    # socket is up (clientThr also retries connect for ~60 s anyway).
    'server_lead_s': 1.0,
    # Drop the leading idle/near-zero prefix of each flow (the period before
    # the TF1 actor is ready and bytes actually flow) and rebase the time
    # axis, so Orca's *active* window aligns with the fixed benchmark window
    # like every other CC. This trims process/model-load latency, not CC
    # convergence behaviour.
    'warmup_trim': True,
    'warmup_trim_eps_mbps': 0.05,
    # Orca prints no RTT; like cctestbed's ss_script.sh we poll `ss -i` on
    # each sender for the kernel srtt of the Orca socket (the same tcpi_rtt
    # Orca itself reads for its RL state). Off => RTT metrics stay NaN.
    'ss_enabled': True,
    'ss_interval_s': 0.1,
}


def _orca_settings(approach: dict, bench: dict) -> dict:
    """Merge per-approach `orca:` overrides on top of the repo defaults.

    `orca_warmup_slack_s` may also be set at the benchmark level.
    """
    cfg = dict(_ORCA_DEFAULTS)
    cfg['warmup_slack_s'] = float(
        bench.get('orca_warmup_slack_s', cfg['warmup_slack_s']))
    overrides = dict(approach.get('orca') or {})
    for key, value in overrides.items():
        if value is not None:
            cfg[key] = value
    cfg['install_dir'] = os.path.abspath(
        os.path.expanduser(str(cfg['install_dir'])))
    return cfg


def _orca_run_user() -> str:
    """The user Orca runs as (it writes into its own rl-module tree and
    expects $HOME/venv/bin/python). Mirrors cctestbed's `sudo -u <USER>`."""
    user = os.environ.get('SUDO_USER')
    if user:
        return user
    try:
        import getpass
        return getpass.getuser()
    except Exception:
        return 'root'


def _orca_user_home(user: str) -> str:
    try:
        import pwd
        return pwd.getpwnam(user).pw_dir
    except Exception:
        home = os.path.expanduser('~' + user)
        if home and not home.startswith('~'):
            return home
        return os.environ.get('HOME', '/root')


def _orca_actor_id(instance_id: int, flow: int) -> int:
    # Orca's SysV-IPC shmem key is actor_id*10000 + rand()%10000, and all
    # mininet hosts share one kernel, so concurrent flows MUST have distinct
    # actor_ids to land in disjoint key ranges. Unique per (slot, flow).
    return int(instance_id) * 100 + int(flow)


def _orca_port(settings: dict, instance_id: int, flow: int) -> int:
    return int(settings['port_base']) + int(instance_id) * 100 + int(flow)


def _orca_paths(settings: dict) -> tuple:
    install = settings['install_dir']
    sender_sh = os.path.join(install, 'sender.sh')
    receiver_sh = os.path.join(install, 'receiver.sh')
    server_bin = os.path.join(install, 'rl-module', 'orca-server-mahimahi')
    client_bin = os.path.join(install, 'rl-module', 'clientThr')
    return install, sender_sh, receiver_sh, server_bin, client_bin


def _run_orca_on_env(env, settings: dict, flow_plan: list, instance_id: int,
                     run_dir: str, duration_s: float) -> None:
    """Launch the real Orca server (c{i}) + clientThr (x{i}) for every flow
    on an already-started MininetEnv, honour per-flow start delays, then
    block until the episode (plus actor-warmup slack) has elapsed.

    This is the `kind: orca` analogue of env.start_episode(): the caller
    builds and starts the env exactly like a kernel trial, calls this, then
    stops the env and parses run_dir/orca_x{f}.log via _orca_receiver_to_csvs().
    """
    install, sender_sh, receiver_sh, server_bin, client_bin = _orca_paths(settings)
    missing = [p for p in (sender_sh, receiver_sh, server_bin, client_bin)
               if not os.path.exists(p)]
    if missing:
        raise SystemExit(
            '[orca] missing Orca install files: ' + ', '.join(missing)
            + f'. Set approach.orca.install_dir (currently {install!r}); the '
            'dir must contain sender.sh, receiver.sh and '
            'rl-module/{orca-server-mahimahi,clientThr} (build with '
            'rl-module build scripts).')

    user = _orca_run_user()
    home = _orca_user_home(user)
    prefix = getattr(env, 'prefix', '') or ''
    os.makedirs(run_dir, exist_ok=True)

    for item in flow_plan:
        flow = int(item['flow'])
        start = max(0.0, float(item.get('start', 0.0)))
        fin = max(1, int(round(float(item.get('duration', duration_s)))))
        actor_id = _orca_actor_id(instance_id, flow)
        port = _orca_port(settings, instance_id, flow)

        c = env.net.get(f'{prefix}c{flow}')
        x = env.net.get(f'{prefix}x{flow}')
        server_ip = c.IP()
        c_log = os.path.join(run_dir, f'orca_c{flow}.log')
        x_log = os.path.join(run_dir, f'orca_x{flow}.log')

        # sender.sh <port> <id> <finish_time> <install>  (period/scheme/max_it
        # are fixed inside sender.sh, exactly as cctestbed runs it).
        server_cmd = (
            f'sudo -u {user} env HOME={home} EXPERIMENT_PATH={run_dir} '
            f'bash {sender_sh} {port} {actor_id} {fin} {install} '
            f'> {c_log} 2>&1')
        # receiver.sh <server_ip> <port> <flow_id> <install>
        client_cmd = (
            f'sudo -u {user} env HOME={home} '
            f'bash {receiver_sh} {server_ip} {port} 0 {install} '
            f'> {x_log} 2>&1')

        if start > 0.0:
            c.cmd(f'(sleep {start:.3f} && {server_cmd}) &')
        else:
            c.cmd(f'{server_cmd} &')
        x.cmd(
            f'(sleep {start + float(settings["server_lead_s"]):.3f} && '
            f'{client_cmd}) &')

        # ss sidecar on the sender: poll the kernel srtt of the Orca data
        # socket (local sport == the server port) every ss_interval_s, epoch
        # prefixed via `ts`. Mirrors cctestbed's ss_script.sh; this is the
        # same tcpi_rtt Orca itself reads via getsockopt(TCP_INFO). Bounded
        # by `timeout` (env.stop() would also kill it at teardown).
        if _as_bool(settings.get('ss_enabled'), default=True):
            ss_log = os.path.join(run_dir, f'orca_ss_c{flow}.log')
            ss_int = max(float(settings.get('ss_interval_s', 0.1)), 0.02)
            ss_to = float(duration_s) + float(settings['warmup_slack_s'])
            ss_loop = (
                f"while :; do ss -tinHO state established "
                f"'( sport = :{port} )' | ts '%.s,'; sleep {ss_int:g}; done")
            c.cmd(
                f'(sleep {start:.3f} && timeout {ss_to:g} '
                f'bash -c "{ss_loop}" > {ss_log} 2>&1) &')

    time.sleep(float(duration_s) + float(settings['warmup_slack_s']))


def _parse_orca_client_output(out_path: str, csv_path: str = None,
                              measure_interval_s: float = 1.0,
                              settings: dict = None) -> dict:
    """Parse a clientThr stdout capture into the same shape _parse_iperf_json
    returns, and (optionally) write the `time,bandwidth` measurement csv.

    clientThr emits, between ----START----/----END----, lines of
    `tick,<mbps>e+06,total_bytes,avg_mbps`. The 2nd field is Mbps with a
    literal `e+06` suffix, so float() yields Mbps*1e6 (mirrors cctestbed's
    parse_orca_output, which divides by 1e6). RTT is unavailable from
    clientThr, so rtt_samples_ms is always empty for Orca.
    """
    settings = settings or _ORCA_DEFAULTS
    try:
        with open(out_path, 'r', errors='replace') as fin:
            blob = fin.read()
    except (FileNotFoundError, OSError):
        return {'goodput_mbps': math.nan, 'samples': [], 'rtt_samples_ms': []}

    start = blob.find('----START----')
    end = blob.find('----END----')
    if start == -1 or end == -1 or end <= start:
        return {'goodput_mbps': math.nan, 'samples': [], 'rtt_samples_ms': []}

    samples = []
    last_avg = math.nan
    for line in blob[start:end].splitlines()[1:]:
        line = line.strip()
        if not line:
            continue
        parts = line.split(',')
        if len(parts) < 2:
            continue
        t_s = _finite_float(parts[0])
        bw = _finite_float(parts[1])          # <mbps>e+06 -> Mbps*1e6
        if not (math.isfinite(t_s) and math.isfinite(bw)):
            continue
        samples.append((t_s, bw / 1e6))
        if len(parts) >= 4:
            avg = _finite_float(parts[3])
            if math.isfinite(avg):
                last_avg = avg

    if samples and (settings.get('warmup_trim', True)):
        eps = float(settings.get('warmup_trim_eps_mbps', 0.05))
        first = next((i for i, (_, v) in enumerate(samples) if v > eps), None)
        if first is not None and first > 0:
            interval = max(float(measure_interval_s or 1.0), 1e-6)
            shift = samples[first][0] - interval
            samples = [(max(t - shift, 0.0), v) for t, v in samples[first:]]

    if csv_path:
        _write_measurement_csv(csv_path, samples, measure_interval_s)

    goodput_mbps = (last_avg if math.isfinite(last_avg)
                    else (sum(v for _, v in samples) / len(samples)
                          if samples else math.nan))
    return {
        'goodput_mbps': goodput_mbps,
        'samples': samples,
        'rtt_samples_ms': [],
    }


def _orca_receiver_to_csvs(run_dir: str, flow_plan: list,
                           measure_interval_s: float = 1.0,
                           settings: dict = None) -> str:
    """Per-flow: parse run_dir/orca_x{f}.log -> run_dir/csvs/x{f}.csv.

    Drop-in replacement for the binned benchmarks' _copy_receiver_iperf_
    outputs(): returns a '; '-joined error string ('' on success).
    """
    csv_dir = os.path.join(run_dir, 'csvs')
    errors = []
    for item in flow_plan:
        flow = int(item['flow'])
        out_path = os.path.join(run_dir, f'orca_x{flow}.log')
        parsed = _parse_orca_client_output(
            out_path, os.path.join(csv_dir, f'x{flow}.csv'),
            measure_interval_s=measure_interval_s, settings=settings)
        if not parsed.get('samples'):
            errors.append(
                f'no Orca clientThr goodput samples for flow {flow} '
                f'({out_path})')
    return '; '.join(errors)


import re as _re


_SS_RTT_RE = _re.compile(r'\brtt:([0-9]+(?:\.[0-9]+)?)')


def _orca_ss_log_path(run_dir: str, flow: int) -> str:
    return os.path.join(run_dir, f'orca_ss_c{int(flow)}.log')


def _parse_orca_ss(ss_path: str, rebase: bool = True) -> list:
    """Parse an `ss -i` poll capture into [(t_s, srtt_ms)].

    Each line is `<epoch>,<ss one-line socket dump…>` (epoch prefixed by
    `ts '%.s,'`); we pull the `rtt:<srtt>/<rttvar>` token (srtt in ms). When
    `rebase`, times are shifted so the first sample sits at ~0 (the socket
    only exists once the flow connects, so this aligns with the goodput
    clock the same way clientThr's tick / warmup-trim do).
    """
    try:
        with open(ss_path, 'r', errors='replace') as fin:
            lines = fin.read().splitlines()
    except (FileNotFoundError, OSError):
        return []

    out = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        head, _, rest = line.partition(',')
        epoch = _finite_float(head)
        if not math.isfinite(epoch):
            continue
        m = _SS_RTT_RE.search(rest)
        if not m:
            continue
        srtt = _finite_float(m.group(1))
        if math.isfinite(srtt) and srtt > 0:
            out.append((epoch, srtt))

    if not out:
        return []
    out.sort(key=lambda s: s[0])
    if rebase:
        t0 = out[0][0]
        out = [(max(t - t0, 0.0), r) for t, r in out]
    return out


def _orca_ss_rtt_samples(run_dir: str, flow_plan: list) -> list:
    """All (episode_time_s, srtt_ms) RTT samples across flows, shifted onto
    the episode clock (per-flow start added) — the Orca analogue of the
    benchmarks' iperf3 _iperf_rtt_samples()."""
    out = []
    for item in flow_plan:
        flow = int(item['flow'])
        start = float(item.get('start', 0.0))
        for t_s, srtt in _parse_orca_ss(_orca_ss_log_path(run_dir, flow)):
            out.append((t_s + start, srtt))
    return out


def _orca_ss_rtt_flows(run_dir: str, flow_plan: list) -> list:
    """Per-flow srtt traces shaped exactly like the benchmarks'
    _iperf_rtt_flows() ({flow,start,end,t,iperf_rtt}) so the existing RTT
    plot panel renders Orca with no plot-code changes."""
    import numpy as _np
    out = []
    for item in flow_plan:
        flow = int(item['flow'])
        start = float(item.get('start', 0.0))
        samples = _parse_orca_ss(_orca_ss_log_path(run_dir, flow))
        t = _np.asarray([s[0] for s in samples], dtype=float) + start
        rtt = _np.asarray([s[1] for s in samples], dtype=float)
        out.append({
            'flow': flow, 'start': start, 'end': float(item.get('end', start)),
            't': t, 'iperf_rtt': rtt,
        })
    return out


# ============================================================================
# Reconstructed convergence/fairness helpers
# ----------------------------------------------------------------------------
# The original `benchmark_convergence/convergence.py` module was deleted.
# These three functions were imported by convergence_4flow / fairness /
# inter_rtt_fairness. They are reconstructed from how the call sites use the
# returned values (no original source was recoverable). Numerical outputs
# may not byte-match the deleted originals; the *shape* and *semantics*
# match what the consumers expect.
# ============================================================================


def _read_iperf_csv(csv_path: str):
    """Load a per-flow iperf3-receiver CSV (time, bandwidth)."""
    t, thr = [], []
    if not os.path.exists(csv_path):
        return np.asarray(t, dtype=float), np.asarray(thr, dtype=float)
    with open(csv_path, newline='') as f:
        for row in csv.DictReader(f):
            try:
                ts = float(row.get('time', ''))
                bw = float(row.get('bandwidth', ''))
            except (TypeError, ValueError):
                continue
            if math.isfinite(ts) and math.isfinite(bw):
                t.append(ts)
                thr.append(bw)
    return np.asarray(t, dtype=float), np.asarray(thr, dtype=float)


def _kernel_flows(run_dir: str, flow_plan: list) -> list:
    """Per-flow throughput series read back from the iperf3-receiver CSVs
    that `_copy_receiver_iperf_outputs` writes under ``run_dir/csvs/x{N}.csv``.

    Returned dicts carry the minimal contract the plotters and the binner
    consume: ``flow``, ``start``, ``end``, ``t`` (seconds since episode
    start), ``thr`` (Mbps). The throughput is whatever `_parse_iperf_json`
    bucketed via `_write_measurement_csv`.

    A flow that joins late is launched via ``(sleep <start> && iperf3 ...)``,
    so iperf3's interval ``end`` timestamps are flow-relative (they start at
    0 when the flow itself begins, not at episode time ``start``). We shift
    them by the flow's scheduled ``start`` here so every series sits on the
    shared episode clock -- this aligns both the individual goodput panels
    and the binned series that feeds the heatmap metrics (a flow with
    ``start == 0`` is unchanged). The RTT traces are shifted by ``start`` the
    same way at their own call sites.
    """
    out = []
    for item in flow_plan:
        flow = int(item['flow'])
        start = float(item.get('start', 0.0))
        end = float(item.get('end', start + float(item.get('duration', 0.0))))
        csv_path = os.path.join(run_dir, 'csvs', f'x{flow}.csv')
        t, thr = _read_iperf_csv(csv_path)
        out.append({
            'flow': flow,
            'start': start,
            'end': end,
            't': t + start if t.size else t,
            'thr': thr,
        })
    return out


def _write_flow_plan(run_dir: str, flow_plan: list) -> str:
    """Persist the per-flow start/duration plan to ``run_dir/flow_plan.csv``.

    Returns the CSV path so callers can record it in the metrics row.
    """
    os.makedirs(run_dir, exist_ok=True)
    path = os.path.join(run_dir, 'flow_plan.csv')
    fields = ['flow', 'start_s', 'duration_s', 'end_s']
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for item in flow_plan:
            start = float(item.get('start', 0.0))
            duration = float(item.get('duration', 0.0))
            end = float(item.get('end', start + duration))
            writer.writerow({
                'flow': int(item['flow']),
                'start_s': f'{start:.3f}',
                'duration_s': f'{duration:.3f}',
                'end_s': f'{end:.3f}',
            })
    return path


def _sample_flow_at(t_query: np.ndarray, flow_t: np.ndarray,
                    flow_thr: np.ndarray) -> np.ndarray:
    """Sample a per-flow throughput series at query times.

    Uses last-known-sample-with-time<=t semantics (step interpolation),
    matching what binned iperf3 buckets represent. Times before the first
    sample yield NaN.
    """
    out = np.full(t_query.shape, np.nan, dtype=float)
    if flow_t.size == 0:
        return out
    idx = np.searchsorted(flow_t, t_query, side='right') - 1
    valid = idx >= 0
    out[valid] = flow_thr[idx[valid]]
    return out


def _binned_series(flows: list, flow_plan: list, bench: dict) -> dict:
    """Per-second aggregate time series across all flows.

    Reconstructed contract (consumed by `_metrics_from_series`,
    `_metrics_from_score_window`, `_goodput_ratio_stats`, and the
    per-run fairness/convergence plotters):

    Output keys (all 1-D arrays of length ``duration_s`` unless noted)
        time          end-of-bin times in seconds, [1, 2, ..., duration]
        per_flow      2-D array (len(flow_plan), duration_s), Mbps
        active_count  number of flows whose [start, end) covers this bin
        total         sum of throughput over active flows
        fair_share    total / max(active_count, 1)
        fair_error    mean over active flows of |thr_i - fair_share|
        jain          Jain's fairness over active flows, NaN if <2 active
    """
    duration = int(float(bench.get('duration_s', 0)) or 0)
    if duration <= 0:
        # Fall back to the largest sample horizon we can see across flows.
        horizons = [float(np.max(np.asarray(f.get('t', []), dtype=float)))
                    for f in flows if len(f.get('t', [])) > 0]
        duration = int(math.ceil(max(horizons))) if horizons else 0
    if duration <= 0:
        empty = np.asarray([], dtype=float)
        return {
            'time': empty, 'per_flow': np.zeros((len(flow_plan), 0)),
            'active_count': empty.astype(int),
            'total': empty, 'fair_share': empty,
            'fair_error': empty, 'jain': empty,
        }

    t_grid = np.arange(1, duration + 1, dtype=float)

    n_plan = len(flow_plan)
    per_flow = np.zeros((n_plan, duration), dtype=float)
    by_flow_id = {int(f.get('flow', i + 1)): f for i, f in enumerate(flows)}
    for plan_idx, item in enumerate(flow_plan):
        flow_id = int(item['flow'])
        flow = by_flow_id.get(flow_id)
        if flow is None:
            continue
        flow_t = np.asarray(flow.get('t', []), dtype=float)
        flow_thr = np.asarray(flow.get('thr', []), dtype=float)
        sampled = _sample_flow_at(t_grid, flow_t, flow_thr)
        # Outside [start, end) the flow is inactive; treat as 0 contribution
        # rather than NaN so totals/sums stay meaningful in the inactive
        # neighbourhood.
        start = float(item.get('start', 0.0))
        end = float(item.get('end', start + float(item.get('duration', 0.0))))
        active_mask = (t_grid >= start) & (t_grid < end)
        sampled = np.where(np.isfinite(sampled), sampled, 0.0)
        per_flow[plan_idx, :] = np.where(active_mask, sampled, 0.0)

    # Active-count mask is from the plan, not from observed samples, so a
    # flow that should be running but has no iperf row still counts as
    # active (its per_flow row is 0 in that bin).
    active_matrix = np.zeros((n_plan, duration), dtype=bool)
    for plan_idx, item in enumerate(flow_plan):
        start = float(item.get('start', 0.0))
        end = float(item.get('end', start + float(item.get('duration', 0.0))))
        active_matrix[plan_idx, :] = (t_grid >= start) & (t_grid < end)

    active_count = active_matrix.sum(axis=0).astype(int)
    total = per_flow.sum(axis=0)
    fair_share = np.where(active_count > 0, total / np.maximum(active_count, 1), 0.0)

    fair_error = np.zeros(duration, dtype=float)
    for bin_idx in range(duration):
        n_active = int(active_count[bin_idx])
        if n_active == 0:
            fair_error[bin_idx] = np.nan
            continue
        active_rows = active_matrix[:, bin_idx]
        thrs = per_flow[active_rows, bin_idx]
        fair_error[bin_idx] = float(np.mean(np.abs(thrs - fair_share[bin_idx])))

    jain = np.full(duration, np.nan, dtype=float)
    for bin_idx in range(duration):
        n_active = int(active_count[bin_idx])
        if n_active < 2:
            continue
        active_rows = active_matrix[:, bin_idx]
        thrs = per_flow[active_rows, bin_idx]
        ssum = float(np.sum(thrs))
        ssq = float(np.sum(thrs * thrs))
        if ssq <= 0.0:
            continue
        jain[bin_idx] = (ssum * ssum) / (n_active * ssq)

    return {
        'time': t_grid,
        'per_flow': per_flow,
        'active_count': active_count,
        'total': total,
        'fair_share': fair_share,
        'fair_error': fair_error,
        'jain': jain,
    }


def _write_rows(path: str, fields: list, rows: list) -> None:
    """Atomically write all rows to a CSV with the given header order."""
    tmp_path = f'{path}.tmp.{os.getpid()}'
    with open(tmp_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, '') for key in fields})
    os.replace(tmp_path, path)

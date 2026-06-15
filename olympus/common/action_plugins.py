"""Action-plugin loading, configuration, and checkpoint compatibility.

Policies emit a normalized action and the selected plugin decides what that
action means at the TCP socket. The orchestrator passes the selected action
name and its YAML options through the environment so learners and workers
resolve the same contract at import time.
"""

import importlib
import json
import os


DEFAULT_ACTION = 'cwnd_multiplier'


def current_action_name(default=DEFAULT_ACTION) -> str:
    name = os.environ.get('SAO_ACTION_NAME', default)
    return str(name or default).strip() or default


def current_action_options() -> dict:
    raw = os.environ.get('SAO_ACTION_CONFIG', '')
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError('SAO_ACTION_CONFIG must be a JSON object') from exc
    if not isinstance(value, dict):
        raise ValueError('SAO_ACTION_CONFIG must be a JSON object')
    return dict(value)


def load_action_module(name: str = None):
    name = str(name or current_action_name()).strip()
    if not name or '.' in name:
        raise ValueError(f'invalid action plugin name: {name!r}')
    return importlib.import_module(f'olympus.actions.{name}')


def action_meta(module=None) -> dict:
    module = module or load_action_module()
    options = dict(module.options())
    return {
        'action_name': str(module.ACTION_NAME),
        'action_dim': int(module.ACTION_DIM),
        'action_version': str(module.ACTION_VERSION),
        'action_options': options,
    }


def checkpoint_action_meta(ckpt: dict) -> dict:
    ckpt = ckpt or {}
    meta = dict(ckpt.get('action_meta') or {})
    if meta:
        return meta
    state = dict(ckpt.get('state_meta') or {})
    return dict(state.get('action_meta') or {})


def assert_action_compatible(current: dict, ckpt: dict,
                             source='checkpoint') -> None:
    saved = checkpoint_action_meta(ckpt)
    if not saved:
        # Checkpoints created before action plugins used the current default
        # scalar log-multiplier mapping.
        if (current.get('action_name') == DEFAULT_ACTION
                and bool(current.get('legacy_default', False))):
            return
        raise ValueError(
            f'{source} has no action metadata, but configured action is '
            f'{current.get("action_name")!r}')

    mismatches = []
    for key in ('action_name', 'action_dim', 'action_version'):
        if key in saved and saved[key] != current.get(key):
            mismatches.append(f'{key}={saved[key]!r} vs {current.get(key)!r}')
    if ('action_options' in saved
            and dict(saved['action_options']) != dict(current.get('action_options') or {})):
        mismatches.append(
            f'action_options={saved["action_options"]!r} vs '
            f'{current.get("action_options")!r}')
    if mismatches:
        raise ValueError(f'{source} action mismatch: ' + '; '.join(mismatches))


def current_action_meta() -> dict:
    module = load_action_module()
    meta = action_meta(module)
    meta['legacy_default'] = bool(
        module.ACTION_NAME == DEFAULT_ACTION
        and getattr(module, 'is_legacy_default', lambda: False)()
    )
    return meta


def action_env(cfg: dict) -> dict:
    runtime = cfg.get('runtime', {}) or {}
    name = str(runtime.get('action', DEFAULT_ACTION))
    options = dict(cfg.get('action', {}) or {})
    return {
        'SAO_ACTION_NAME': name,
        'SAO_ACTION_CONFIG': json.dumps(
            options, sort_keys=True, separators=(',', ':')),
    }


__all__ = [
    'DEFAULT_ACTION',
    'action_env',
    'action_meta',
    'assert_action_compatible',
    'checkpoint_action_meta',
    'current_action_meta',
    'current_action_name',
    'current_action_options',
    'load_action_module',
]

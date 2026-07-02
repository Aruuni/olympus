"""Optional raw observation logging for worker flow backends."""

import json
import math
import os
import time

_LOG_FILE = None
_LOG_PATH = None
_START_WALL = None


def _as_bool(value, default=False):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in ('1', 'true', 'yes', 'on'):
        return True
    if text in ('0', 'false', 'no', 'off', ''):
        return False
    return default


def _enabled():
    return _as_bool(os.environ.get('SAO_RAW_STATE_LOG_ENABLED'), False)


def _suffix_path(path):
    flow_id = os.environ.get('OC_FLOW_ID') or os.environ.get('SAO_AGENT_ID')
    if flow_id is None or flow_id == '':
        flow_id = '0'
    root, ext = os.path.splitext(path)
    ext = ext or '.jsonl'
    return f'{root}_flow{flow_id}{ext}'


def _open_log():
    global _LOG_FILE, _LOG_PATH, _START_WALL
    if _LOG_FILE is not None:
        return _LOG_FILE
    base = os.environ.get('SAO_RAW_STATE_LOG', '')
    if not base:
        return None
    _LOG_PATH = _suffix_path(base)
    os.makedirs(os.path.dirname(os.path.abspath(_LOG_PATH)), exist_ok=True)
    _LOG_FILE = open(_LOG_PATH, 'a', buffering=1)
    _START_WALL = time.monotonic()
    return _LOG_FILE


def _jsonable(value):
    try:
        if hasattr(value, 'item'):
            value = value.item()
    except Exception:
        pass
    if isinstance(value, (bool, int, str)) or value is None:
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return str(value)


def record(raw):
    """Append one raw observation as JSONL when logging is enabled."""
    if not _enabled() or not isinstance(raw, dict):
        return
    handle = _open_log()
    if handle is None:
        return
    wall_now = time.monotonic()
    wall_start = _START_WALL if _START_WALL is not None else wall_now
    row = {str(key): _jsonable(value) for key, value in raw.items()}
    row.setdefault('_wall_t_s', wall_now - wall_start)
    row.setdefault('_backend', os.environ.get('OC_FLOW_BACKEND', 'tcp'))
    row.setdefault('_flow_id', os.environ.get('OC_FLOW_ID', ''))
    row.setdefault('_agent_id', os.environ.get('SAO_AGENT_ID', ''))
    row.setdefault('_cport', os.environ.get('OC_CPORT', ''))
    row.setdefault('_pid', os.getpid())
    try:
        handle.write(json.dumps(row, sort_keys=True) + '\n')
    except Exception:
        pass

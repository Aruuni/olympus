"""Small import registry for hot-swappable algorithms, rewards, and actions.

One flat namespace holds both single-agent algorithms (td3, orca, dreamer_v3,
…) and multi-agent ones (ma_dreamer, ma_td3). Each algorithm package declares its
nature with a module-level ``MULTI_AGENT`` flag in ``algorithms/<name>/__init__
.py`` (default ``False``). The orchestrator reads that flag via
``is_multi_agent`` to decide between true multi-agent rollouts and single-agent
lagged self-play in multi-flow environments.
"""

import importlib
import importlib.util


_BASE = 'olympus'


def _script_path(module_name: str) -> str:
    """Resolve a module's file path without importing it.

    Used for learner/worker scripts so the orchestrator parent can learn the
    path it will launch as a subprocess without importing the (torch-heavy)
    module into its own process.
    """
    spec = importlib.util.find_spec(module_name)
    if spec is None or not spec.origin:
        raise ModuleNotFoundError(f'module {module_name!r} was not found')
    return spec.origin


def algorithm_module(name: str):
    """Return the algorithm package selected by config."""
    return importlib.import_module(f'{_BASE}.algorithms.{name}')


def is_multi_agent(name: str) -> bool:
    """Return True if the selected algorithm's learner supports multi-agent.

    Read from the ``MULTI_AGENT`` flag on ``algorithms/<name>/__init__.py``.
    Importing the package only runs its (docstring-only) ``__init__``, so this
    is cheap and does not pull in torch. Defaults to single-agent when the flag
    is absent.
    """
    return bool(getattr(algorithm_module(name), 'MULTI_AGENT', False))


def model_module(name: str):
    """Return the selected algorithm's model module."""
    return importlib.import_module(f'{_BASE}.algorithms.{name}.model')


def learner_script(name: str) -> str:
    """Return a filesystem path for the selected algorithm's learner script."""
    return _script_path(f'{_BASE}.algorithms.{name}.learner')


def worker_script(name: str) -> str:
    """Return the selected algorithm worker script.

    Every algorithm must provide algorithms/<name>/worker.py so rollout
    behavior is explicit and versioned with the algorithm.
    """
    return _script_path(f'{_BASE}.algorithms.{name}.worker')


def reward_module(name: str):
    """Return the selected reward plugin module."""
    return importlib.import_module(f'{_BASE}.rewards.{name}')


def action_module(name: str):
    """Return the selected action plugin module."""
    return importlib.import_module(f'{_BASE}.actions.{name}')

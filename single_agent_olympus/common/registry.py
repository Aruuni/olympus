"""Small import registry for hot-swappable algorithms and rewards."""

import importlib


_BASE = 'single_agent_olympus'


def algorithm_module(name: str):
    """Return the algorithm package selected by config."""
    return importlib.import_module(f'{_BASE}.algorithms.{name}')


def model_module(name: str):
    """Return the selected algorithm's model module."""
    return importlib.import_module(f'{_BASE}.algorithms.{name}.model')


def learner_script(name: str) -> str:
    """Return a filesystem path for the selected algorithm's learner script."""
    mod = importlib.import_module(f'{_BASE}.algorithms.{name}.learner')
    return mod.__file__


def worker_script(name: str) -> str:
    """Return the selected algorithm worker script.

    Every algorithm must provide algorithms/<name>/worker.py so rollout
    behavior is explicit and versioned with the algorithm.
    """
    mod = importlib.import_module(f'{_BASE}.algorithms.{name}.worker')
    return mod.__file__


def reward_module(name: str):
    """Return the selected reward plugin module."""
    return importlib.import_module(f'{_BASE}.rewards.{name}')

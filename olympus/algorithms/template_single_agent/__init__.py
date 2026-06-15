"""Single-agent algorithm template.

Copy this directory to ``algorithms/<your_algorithm>/`` and rename it, then fill
in ``model.py``. The orchestrator selects this package via ``runtime.algorithm``
in config.yaml and reads ``MULTI_AGENT`` from here through
``common.registry.is_multi_agent`` to choose the single- vs multi-agent rollout
path. Leave it False for a single-agent algorithm (any environment with >1 flow
becomes lagged self-play automatically; one worker process per flow).
"""

MULTI_AGENT = False

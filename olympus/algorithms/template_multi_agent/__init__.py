"""Multi-agent algorithm template.

Copy this directory to ``algorithms/<your_algorithm>/`` and rename it, then fill
in ``model.py``. The ``MULTI_AGENT = True`` flag is what ``common.registry.is_multi_agent``
reads to send the orchestrator down the multi-agent rollout path
(``run_episode_marl``): every flow in the ``sweep`` block is trained jointly, the
orchestrator launches one listener+worker per flow, and the learner reconstructs
the joint state from the per-flow experiences. Leave it True.
"""

MULTI_AGENT = True  # learner trains every flow jointly (true multi-agent)

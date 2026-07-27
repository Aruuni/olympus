"""Checkpoint-backed policy owned by exactly one flow process.

Algorithm-specific construction lives with each algorithm, behind the
`build_policy` factory documented in olympus/common/policy.py. This module only
resolves config, binds the state/action plugins the model modules read at
import time, and adapts the resulting Policy to the worker's `infer` call — so
a new algorithm deploys as soon as it ships a `build_policy`, with no edit
here.
"""

import importlib
import json
import os


def _selected_config(config, algorithm):
    """Merge raw algorithm blocks with authoritative resolved top-level blocks."""
    selected = ((config.get("algorithms", {}) or {}).get(algorithm, {}) or {})
    agent = dict(selected.get("agent", {}) or {})
    agent.update(config.get("agent", {}) or {})
    training = dict(selected.get("training", {}) or {})
    training.update(config.get("training", {}) or {})
    return agent, training


class FlowPolicy:
    """Load one actor and retain its state for this process's only flow."""

    def __init__(self, config, checkpoint, device="cpu"):
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        import torch

        self.torch = torch
        self.device = torch.device(device)
        runtime = config.get("runtime", {}) or {}
        self.algorithm = str(runtime.get("algorithm", "td3"))
        agent, training = _selected_config(config, self.algorithm)

        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        checkpoint_state = dict(ckpt.get("state_meta") or {})
        self.state_name = str(
            checkpoint_state.get("state_name")
            or runtime.get("state", "default_orca"))

        # Model modules bind their state/action plugins at import time, so the
        # environment has to be set before the import below.
        action_name = str(runtime.get("action", "cwnd_multiplier"))
        action_options = (config.get("actions", {}) or {}).get(
            action_name, {}) or {}
        os.environ["SAO_ACTION_NAME"] = action_name
        os.environ["SAO_ACTION_CONFIG"] = json.dumps(action_options)
        os.environ["SAO_STATE"] = self.state_name
        os.environ["SAO_STATE_CONFIG"] = json.dumps(
            config.get("state_options", {}) or {})
        try:
            self.module = importlib.import_module(
                f"olympus.algorithms.{self.algorithm}.model")
        except ImportError as exc:
            raise ValueError(
                f"unknown algorithm {self.algorithm!r}: no "
                f"olympus.algorithms.{self.algorithm}.model") from exc
        if not hasattr(self.module, "build_policy"):
            raise ValueError(
                f"algorithm {self.algorithm!r} cannot be deployed: its model "
                "module defines no build_policy(); see olympus/common/policy.py")

        self.deterministic = bool((config.get("deployment", {}) or {}).get(
            "deterministic", True))
        # build_policy registers checkpoint-borne architecture (ma_td3's
        # history/normalization) that the compatibility check validates
        # against, so it has to run first.
        self.policy = self.module.build_policy(
            ckpt, agent, training, device=device,
            deterministic=self.deterministic)
        if hasattr(self.module, "assert_checkpoint_state_compatible"):
            self.module.assert_checkpoint_state_compatible(ckpt, checkpoint)

    @property
    def state_dim(self):
        return int(self.module.STATE_DIM)

    def reset(self):
        """Drop carried state (LSTM/GRU cells, RSSM latents, history)."""
        self.policy.reset()

    def infer(self, state):
        return self.policy.act(state)

"""Checkpoint-backed policy owned by exactly one flow process."""

import importlib
import json
import os

import numpy as np


_SUPPORTED = {"td3", "mbpo_td3", "dreamer_v3", "ma_dreamer"}


def _selected_config(config, algorithm):
    """Merge raw algorithm blocks with authoritative resolved top-level blocks."""
    selected = ((config.get("algorithms", {}) or {}).get(algorithm, {}) or {})
    agent = dict(selected.get("agent", {}) or {})
    agent.update(config.get("agent", {}) or {})
    training = dict(selected.get("training", {}) or {})
    training.update(config.get("training", {}) or {})
    return agent, training


class FlowPolicy:
    """Load one actor and retain recurrent state for this process's only flow."""

    def __init__(self, config, checkpoint, device="cpu"):
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        import torch

        self.torch = torch
        self.device = torch.device(device)
        runtime = config.get("runtime", {}) or {}
        self.algorithm = str(runtime.get("algorithm", "td3"))
        if self.algorithm not in _SUPPORTED:
            raise ValueError(
                "per-flow deployment supports td3, mbpo_td3, dreamer_v3, "
                f"and ma_dreamer; got {self.algorithm!r}")
        agent, training = _selected_config(config, self.algorithm)

        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        checkpoint_state = dict(ckpt.get("state_meta") or {})
        self.state_name = str(
            checkpoint_state.get("state_name")
            or runtime.get("state", "default_orca"))

        action_name = str(runtime.get("action", "cwnd_multiplier"))
        action_options = (config.get("actions", {}) or {}).get(
            action_name, {}) or {}
        os.environ["SAO_ACTION_NAME"] = action_name
        os.environ["SAO_ACTION_CONFIG"] = json.dumps(action_options)
        os.environ["SAO_STATE"] = self.state_name
        os.environ["SAO_STATE_CONFIG"] = json.dumps(
            config.get("state_options", {}) or {})
        self.module = importlib.import_module(
            f"olympus.algorithms.{self.algorithm}.model")
        if hasattr(self.module, "assert_checkpoint_state_compatible"):
            self.module.assert_checkpoint_state_compatible(ckpt, checkpoint)

        self.deterministic = bool((config.get("deployment", {}) or {}).get(
            "deterministic", True))
        self.recurrent = None
        if self.algorithm in {"dreamer_v3", "ma_dreamer"}:
            self._init_dreamer(agent, training, ckpt)
        else:
            self._init_actor(agent, ckpt)

    def _init_actor(self, agent, ckpt):
        hidden = int(agent.get("hidden", 128))
        head = agent.get("head_hidden")
        head = int(head) if head not in (None, "") else None
        if hasattr(self.module, "actor_arch_from_checkpoint"):
            hidden, head = self.module.actor_arch_from_checkpoint(
                ckpt, hidden, head)
        self.actor = self.module.Actor(self.module.STATE_DIM, hidden, head)
        self.actor.load_state_dict(
            ckpt.get("actor", ckpt.get("actor_state_dict")))
        self.actor.to(self.device).eval()
        self.hidden_size = int(hidden)
        self._dreamer = False

    def _init_dreamer(self, agent, training, ckpt):
        hidden = int(agent.get("hidden", 256))
        embed = int(agent.get("embed_dim", 128))
        h_dim = int(agent.get("h_dim", 256))
        groups = int(agent.get("latent_groups", 8))
        classes = int(agent.get("latent_classes", 8))
        bins = int(agent.get("reward_bins", 255))
        if self.algorithm == "ma_dreamer":
            self.world = self.module.LocalWorldModel(
                self.module.STATE_DIM, self.module.ACTION_DIM, hidden, embed,
                h_dim, groups, classes, bins)
            world_key = "local_world_model"
        else:
            self.world = self.module.WorldModel(
                self.module.STATE_DIM, self.module.ACTION_DIM, hidden, embed,
                h_dim, groups, classes, bins)
            world_key = "world_model"
        self.actor = self.module.Actor(
            h_dim, self.world.latent_dim, hidden,
            log_std_min=float(training.get("actor_log_std_min", -2.0)),
            log_std_max=float(training.get("actor_log_std_max", 0.0)))
        self.world.load_state_dict(ckpt[world_key], strict=False)
        self.actor.load_state_dict(ckpt["actor"], strict=False)
        self.world.to(self.device).eval()
        self.actor.to(self.device).eval()
        self._dreamer = True

    @property
    def state_dim(self):
        return int(self.module.STATE_DIM)

    def infer(self, state):
        state = np.asarray(state, dtype=np.float32).reshape(-1)
        if state.shape != (self.state_dim,):
            raise ValueError(
                f"expected state shape ({self.state_dim},), got {state.shape}")
        if self._dreamer:
            return self._infer_dreamer(state)
        return self._infer_actor(state)

    def _infer_actor(self, state):
        torch = self.torch
        if self.recurrent is None:
            h = (
                torch.zeros(1, 1, self.hidden_size, device=self.device),
                torch.zeros(1, 1, self.hidden_size, device=self.device),
            )
        else:
            h = self.recurrent
        tensor = torch.as_tensor(
            state, device=self.device).reshape(1, 1, self.state_dim)
        with torch.inference_mode():
            actions, h_new = self.actor.forward_sequence(tensor, h)
        self.recurrent = (h_new[0].detach(), h_new[1].detach())
        return float(actions[0, -1].detach().cpu().reshape(-1)[0])

    def _infer_dreamer(self, state):
        torch = self.torch
        if self.recurrent is None:
            h, z = self.world.rssm.initial(1, self.device)
            previous = torch.zeros(
                1, self.module.ACTION_DIM, device=self.device)
        else:
            h, z, previous = self.recurrent
        tensor = torch.as_tensor(
            state, device=self.device).reshape(1, self.state_dim)
        with torch.inference_mode():
            embedding = self.world.encoder(tensor)
            h, z, _, _ = self.world.rssm.step(
                h, z, previous, embedding,
                sample=not self.deterministic)
            action, _, _, _ = self.actor.act(
                h, z, deterministic=self.deterministic)
        self.recurrent = (h.detach(), z.detach(), action.detach())
        return float(action.detach().cpu().reshape(-1)[0])


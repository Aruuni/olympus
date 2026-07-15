"""Checkpoint-backed recurrent dynamic-batch policy adapter."""

import importlib
import json
import os

import numpy as np


def _selected_config(config, algorithm):
    selected = ((config.get("algorithms", {}) or {}).get(algorithm, {}) or {})
    agent = dict(selected.get("agent", {}) or {})
    agent.update(config.get("agent", {}) or {})
    training = dict(selected.get("training", {}) or {})
    training.update(config.get("training", {}) or {})
    return agent, training


class BatchPolicy:
    """Load an Olympus actor once and carry recurrent state per flow."""

    def __init__(self, config, checkpoint, device="cpu"):
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        import torch

        self.torch = torch
        self.device = torch.device(device)
        runtime = config.get("runtime", {}) or {}
        self.algorithm = str(runtime.get("algorithm", "td3"))
        if self.algorithm not in {"td3", "mbpo_td3", "dreamer_v3", "ma_dreamer"}:
            raise ValueError(
                "deployment batching supports td3, mbpo_td3, dreamer_v3, "
                "and ma_dreamer; "
                f"got {self.algorithm!r}")
        agent_cfg, training_cfg = _selected_config(config, self.algorithm)
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        checkpoint_state = dict(ckpt.get("state_meta") or {})
        self.state_name = str(
            checkpoint_state.get("state_name")
            or runtime.get("state", "default_orca"))
        # Model modules bind their state/action plugins at import time. The
        # checkpoint contract is authoritative; resolved configs can be stale
        # or, historically, ignored by an algorithm implementation.
        action_name = str(runtime.get("action", "cwnd_multiplier"))
        action_options = (config.get("actions", {}) or {}).get(
            action_name, {}) or {}
        os.environ["SAO_ACTION_NAME"] = action_name
        os.environ["SAO_ACTION_CONFIG"] = json.dumps(action_options)
        os.environ["SAO_STATE"] = self.state_name
        os.environ["SAO_STATE_CONFIG"] = json.dumps(config.get("state_options", {}) or {})
        self.module = importlib.import_module(
            f"olympus.algorithms.{self.algorithm}.model")
        if hasattr(self.module, "assert_checkpoint_state_compatible"):
            self.module.assert_checkpoint_state_compatible(ckpt, checkpoint)
        self.deterministic = bool((config.get("deployment", {}) or {}).get(
            "deterministic", True))
        if self.algorithm in {"dreamer_v3", "ma_dreamer"}:
            self._init_dreamer(agent_cfg, training_cfg, ckpt)
            return
        hidden = int(agent_cfg.get("hidden", 128))
        head = agent_cfg.get("head_hidden")
        head = int(head) if head not in (None, "") else None
        if hasattr(self.module, "actor_arch_from_checkpoint"):
            hidden, head = self.module.actor_arch_from_checkpoint(ckpt, hidden, head)
        self.actor = self.module.Actor(self.module.STATE_DIM, hidden, head)
        self.actor.load_state_dict(ckpt.get("actor", ckpt.get("actor_state_dict")))
        self.actor.to(self.device).eval()
        self.hidden_size = int(hidden)
        self.hidden = {}

    def _init_dreamer(self, agent_cfg, training, ckpt):
        hidden = int(agent_cfg.get("hidden", 256))
        embed = int(agent_cfg.get("embed_dim", 128))
        h_dim = int(agent_cfg.get("h_dim", 256))
        groups = int(agent_cfg.get("latent_groups", 8))
        classes = int(agent_cfg.get("latent_classes", 8))
        bins = int(agent_cfg.get("reward_bins", 255))
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
        self.hidden = {}
        self._dreamer = True

    @property
    def state_dim(self):
        return int(self.module.STATE_DIM)

    def normalize(self, raw):
        return self.module.normalize_state(raw)

    def detach(self, flow_id):
        self.hidden.pop(str(flow_id), None)

    def infer(self, requests):
        if getattr(self, "_dreamer", False):
            return self._infer_dreamer(requests)
        torch = self.torch
        states = np.asarray([
            r["state"] if "state" in r else self.normalize(r["raw"])
            for r in requests
        ], dtype=np.float32)
        if states.shape != (len(requests), self.state_dim):
            raise ValueError(f"expected state shape (*,{self.state_dim}), got {states.shape}")
        keys = [str(r["flow_id"]) for r in requests]
        h_parts, c_parts = [], []
        for key in keys:
            pair = self.hidden.get(key)
            if pair is None:
                h_parts.append(torch.zeros(1, 1, self.hidden_size, device=self.device))
                c_parts.append(torch.zeros(1, 1, self.hidden_size, device=self.device))
            else:
                h_parts.append(pair[0])
                c_parts.append(pair[1])
        h = (torch.cat(h_parts, dim=1), torch.cat(c_parts, dim=1))
        tensor = torch.as_tensor(states, device=self.device).unsqueeze(1)
        with torch.inference_mode():
            actions, h_new = self.actor.forward_sequence(tensor, h)
        values = actions[:, -1].detach().cpu().numpy()
        for i, key in enumerate(keys):
            self.hidden[key] = (
                h_new[0][:, i:i + 1].detach(), h_new[1][:, i:i + 1].detach())
        return [float(np.asarray(v).reshape(-1)[0]) for v in values]

    def _infer_dreamer(self, requests):
        """Batch independent local RSSM states for Dreamer and MA-Dreamer.

        MA-Dreamer's global world model is training-only. Its checkpoint's
        local world model and shared actor implement decentralized execution,
        exactly matching ma_dreamer/worker.py.
        """
        torch = self.torch
        states = np.asarray([
            r["state"] if "state" in r else self.normalize(r["raw"])
            for r in requests
        ], dtype=np.float32)
        keys = [str(r["flow_id"]) for r in requests]
        hs, zs, previous = [], [], []
        for key in keys:
            value = self.hidden.get(key)
            if value is None:
                h, z = self.world.rssm.initial(1, self.device)
                a = torch.zeros(1, self.module.ACTION_DIM, device=self.device)
            else:
                h, z, a = value
            hs.append(h); zs.append(z); previous.append(a)
        h = torch.cat(hs, dim=0)
        z = torch.cat(zs, dim=0)
        a_prev = torch.cat(previous, dim=0)
        state = torch.as_tensor(states, device=self.device)
        with torch.inference_mode():
            embedding = self.world.encoder(state)
            h, z, _, _ = self.world.rssm.step(
                h, z, a_prev, embedding, sample=not self.deterministic)
            action, _, _, _ = self.actor.act(
                h, z, deterministic=self.deterministic)
        for index, key in enumerate(keys):
            self.hidden[key] = (
                h[index:index + 1].detach(), z[index:index + 1].detach(),
                action[index:index + 1].detach())
        return [float(v) for v in action.reshape(len(requests), -1)[:, 0].cpu()]

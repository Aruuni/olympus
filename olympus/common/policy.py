"""Uniform inference contract shared by the deployment services.

A training worker owns exploration noise, replay pushing, reward computation,
episode bookkeeping and checkpoint hot-reload. A deployment service needs only
the small part in the middle: load a checkpoint, then map one normalized
observation to one bounded action. Every algorithm's model module exposes that
as a single factory::

    build_policy(ckpt, agent_cfg=None, training_cfg=None,
                 device='cpu', deterministic=True) -> Policy

``Policy.act(state)`` returns exactly the value the algorithm's worker hands to
``action_plugin.apply_cwnd`` — bounded in [-1, 1] for every current algorithm —
and ``Policy.reset()`` drops whatever per-flow state the policy carries (LSTM
or GRU cells, RSSM latents, an observation history window).

The adapters below are keyed on *calling convention* rather than on algorithm,
so a new algorithm normally picks an existing one instead of writing another.
Which convention an algorithm uses is otherwise undiscoverable: actors differ
in constructor signature, in whether they carry recurrent state, in whether
that state is a tensor or a tuple, and in how many values ``act`` returns.
"""

import numpy as np
import torch


class Policy:
    """Base class: shape validation, state reset, and a stable public API."""

    def __init__(self, algorithm, state_dim, deterministic=True):
        self.algorithm = str(algorithm)
        self.state_dim = int(state_dim)
        self.deterministic = bool(deterministic)
        self.reset()

    def reset(self):
        """Forget per-flow state. Safe to call before the first act()."""

    def act(self, state):
        """One normalized observation -> one bounded action."""
        prepared = np.asarray(state, dtype=np.float32).reshape(-1)
        if prepared.shape != (self.state_dim,):
            raise ValueError(
                f"{self.algorithm} policy expected state shape "
                f"({self.state_dim},), got {prepared.shape}")
        return float(self._act(prepared))

    def _act(self, state):
        raise NotImplementedError


class RecurrentActorPolicy(Policy):
    """actor.act(state, h, noise_std=0.0) -> (action, mult, h). TD3-style."""

    def __init__(self, algorithm, state_dim, actor, deterministic=True):
        self.actor = actor
        super().__init__(algorithm, state_dim, deterministic)

    def reset(self):
        self.hidden = None

    def _act(self, state):
        action, _mult, self.hidden = self.actor.act(
            state, self.hidden, noise_std=0.0)
        return action


class StochasticRecurrentActorPolicy(Policy):
    """actor.act(state, h, deterministic=) -> (action, mult, log_std, h). SAC."""

    def __init__(self, algorithm, state_dim, actor, deterministic=True):
        self.actor = actor
        super().__init__(algorithm, state_dim, deterministic)

    def reset(self):
        self.hidden = None

    def _act(self, state):
        action, _mult, _log_std, self.hidden = self.actor.act(
            state, self.hidden, deterministic=self.deterministic)
        return action


class StatelessActorPolicy(Policy):
    """actor.act(state, noise_std=0.0) -> (action, mult). No carried state."""

    def __init__(self, algorithm, state_dim, actor, deterministic=True):
        self.actor = actor
        super().__init__(algorithm, state_dim, deterministic)

    def _act(self, state):
        action, _mult = self.actor.act(state, noise_std=0.0)
        return action


class StackedHistoryPolicy(Policy):
    """actor.act(window, noise_std=0.0) over a rolling frame window.

    The window starts as zeros and shifts one observation in per step, which is
    the ``rec_buffer`` / ``s0_rec_buffer`` convention of the Astraea actor.
    """

    def __init__(self, algorithm, state_dim, actor, history_len,
                 deterministic=True):
        self.actor = actor
        self.history_len = int(history_len)
        super().__init__(algorithm, state_dim, deterministic)

    def reset(self):
        self.window = np.zeros(
            self.history_len * self.state_dim, dtype=np.float32)

    def _act(self, state):
        self.window = np.concatenate([self.window[self.state_dim:], state])
        action, _mult = self.actor.act(self.window, noise_std=0.0)
        return action


class TanhGaussianRecurrentPolicy(Policy):
    """net.act(state, h, deterministic=) -> (raw, mult, ..., h, ...). PPO.

    The net returns a pre-tanh action; the worker squashes it before applying
    it to cwnd, so this adapter does the same.
    """

    def __init__(self, algorithm, state_dim, net, hidden_index=4,
                 deterministic=True):
        self.net = net
        self.hidden_index = int(hidden_index)
        super().__init__(algorithm, state_dim, deterministic)

    def reset(self):
        self.hidden = None

    def _act(self, state):
        outputs = self.net.act(
            state, self.hidden, deterministic=self.deterministic)
        self.hidden = outputs[self.hidden_index]
        return float(np.tanh(outputs[0]))


class LatentWorldModelPolicy(Policy):
    """Dreamer: encoder -> rssm.step -> actor.act(h, z).

    ``deterministic`` selects the actor's mean action but does NOT switch the
    RSSM posterior to its mode: the worker always samples the latent, and the
    policy was trained under that distribution.
    """

    def __init__(self, algorithm, state_dim, world, actor, action_dim,
                 device="cpu", deterministic=True):
        self.world = world
        self.actor = actor
        self.action_dim = int(action_dim)
        self.device = torch.device(device)
        super().__init__(algorithm, state_dim, deterministic)

    def reset(self):
        self.latent = None

    def _act(self, state):
        if self.latent is None:
            h, z = self.world.rssm.initial(1, self.device)
            previous = torch.zeros(1, self.action_dim, device=self.device)
        else:
            h, z, previous = self.latent
        tensor = torch.as_tensor(
            state, device=self.device).reshape(1, self.state_dim)
        with torch.inference_mode():
            embedding = self.world.encoder(tensor)
            h, z, _, _ = self.world.rssm.step(h, z, previous, embedding)
            action, _mult, _mu, _log_std = self.actor.act(
                h, z, deterministic=self.deterministic)
        self.latent = (h.detach(), z.detach(), action.detach())
        return float(action.detach().cpu().reshape(-1)[0])


def actor_state_dict(ckpt, *keys):
    """First present actor payload among `keys`, defaulting to the usual two."""
    for key in keys or ("actor", "actor_state_dict"):
        payload = (ckpt or {}).get(key)
        if payload is not None:
            return payload
    raise KeyError(
        f"checkpoint has no actor weights under {keys or ('actor',)}")


def resolved_hidden(agent_cfg, key, default):
    value = (agent_cfg or {}).get(key, default)
    return int(default if value in (None, "") else value)

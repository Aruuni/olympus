"""
model.py — PROVIDE YOUR OWN.

This file is intentionally a stub. The worker and learner in this template are
generic scaffolding; the actual algorithm (networks, action selection, losses)
lives here and is the one part you must write. See algorithms/td3/model.py for a
complete, working reference implementation.

The worker and learner in this template import the symbols listed below. As long
as you expose this contract, you can implement the internals however your
algorithm requires (TD3, SAC, PPO, a world model, …). Delete the NotImplemented
guard at the bottom once you have implemented everything.

────────────────────────────────────────────────────────────────────────────
REQUIRED MODULE-LEVEL CONSTANTS
────────────────────────────────────────────────────────────────────────────
  STATE_DIM   : int   — observation dimensionality. In this codebase it is
                        normally sourced from the selected state plugin, e.g.
                            _STATE_PLUGIN = load_state_module('<your_alg>')
                            STATE_DIM = int(_STATE_PLUGIN.STATE_DIM)
  ACTION_DIM  : int   — action dimensionality (usually from the action plugin:
                            _ACTION_PLUGIN = load_action_module()
                            ACTION_DIM = int(_ACTION_PLUGIN.ACTION_DIM))
  ACTION_MIN  : float — bounds of the (squashed) action the worker stores.
  ACTION_MAX  : float

  Experience  : a namedtuple. The worker constructs these and pushes them to the
                learner; the learner's replay buffer consumes them. The field
                set is yours to choose, but the template worker/learner assume:
                    Experience = namedtuple('Experience',
                        ['state', 'action', 'reward', 'next_state', 'done',
                         'traj_id', 'step_in_traj'])
                ``traj_id`` MUST be formatted '<algorithm>_<cport>_<episode>_<flow>'
                so the learner can recover the episode index for per-episode
                checkpointing (see _episode_from_traj_id in the learner).

────────────────────────────────────────────────────────────────────────────
REQUIRED FUNCTIONS
────────────────────────────────────────────────────────────────────────────
  normalize_state(info: dict) -> np.ndarray
      Map one tcp_sockopt.get_tcp_deepcc_info() reading (plus the worker-injected
      history keys prev_urtt / prev_cwnd / peak_thr / interval_ms) into a
      length-STATE_DIM float32 observation. Typically delegated to the state
      plugin: normalize_state = _STATE_PLUGIN.normalize_state

  model_state_meta() -> dict
      Identity of the state representation, embedded in checkpoints so a worker
      can refuse to load a checkpoint trained on a different state. Usually:
          return state_meta(_STATE_PLUGIN, STATE_NAME)

  assert_checkpoint_state_compatible(ckpt: dict, source='checkpoint') -> None
      Raise if ckpt's stored state_meta disagrees with model_state_meta().

────────────────────────────────────────────────────────────────────────────
REQUIRED CLASSES / POLICY API
────────────────────────────────────────────────────────────────────────────
  Actor(STATE_DIM, hidden, head_hidden) : nn.Module
      The deployable policy. Must implement:
        act(state_np, hidden_state=None, noise_std=0.0)
            -> (action_float, mult_float, new_hidden_state)
            action_float : the bounded action stored in replay (∈ [ACTION_MIN,
                           ACTION_MAX]); mult_float : value passed to the action
            plugin / logged; new_hidden_state : carried across steps (or None
            for a feed-forward policy).
      The learner additionally uses a training-time forward + your critic and
      loss functions. The exact set depends on your algorithm; for an off-policy
      actor-critic the td3 reference exposes:
        TwinCritic(STATE_DIM, hidden, head_hidden)
        critic_loss(...) -> CriticInfo
        actor_loss(...)  -> ActorInfo
        soft_update(online, target, tau)
        actor_arch_from_checkpoint(ckpt, hidden, head_hidden) -> (hidden, head_hidden)
        actor_model_meta(actor) -> dict

Wire STATE_DIM/ACTION_DIM through the shared plugins so rollout data stays
comparable across algorithms — do not hardcode the dimensionality.
"""

# ── Implement the contract above, then delete this guard. ─────────────────────
raise NotImplementedError(
    'template_single_agent/model.py is a stub — copy this package, rename it, '
    'and implement model.py. See algorithms/td3/model.py for a reference.'
)

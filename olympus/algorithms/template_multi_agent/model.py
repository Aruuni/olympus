"""
model.py — PROVIDE YOUR OWN.

This file is intentionally a stub. The worker and learner in this template are
generic multi-agent scaffolding; the actual algorithm (the joint/parameter-shared
policy, the centralized critic, the losses) lives here and is the one part you
must write. See algorithms/ma_td3/model.py (off-policy CTDE) or
algorithms/ma_dreamer/model.py (off-policy world model) for complete references.

────────────────────────────────────────────────────────────────────────────
WHAT "MULTI-AGENT" MEANS HERE
────────────────────────────────────────────────────────────────────────────
The orchestrator runs N flows on the same bottleneck simultaneously and launches
one worker per flow (decentralized execution). Each worker tags its experiences
with:
    agent_id   — this flow's index within the slot (0 .. n_agents-1)
    group_id   — 'slot{instance}_ep{episode}', identical for all flows sharing a
                 bottleneck in one episode
    group_step — the slot-wide control tick (round(t_s / interval_s))
The learner regroups experiences by (group_id, group_step) so each timestep
yields ONE joint/centralized training sample across the simultaneous flows. This
is where the fairness/team reward is computed from sibling throughputs.

────────────────────────────────────────────────────────────────────────────
REQUIRED MODULE-LEVEL CONSTANTS
────────────────────────────────────────────────────────────────────────────
  STATE_DIM   : int   — per-agent observation dimensionality (from the state
                        plugin, same as the single-agent contract).
  ACTION_DIM  : int   — per-agent action dimensionality (from the action plugin).

  Experience  : a namedtuple. The MAT (on-policy CTDE) field set the template
                worker/learner assume is:
                    Experience = namedtuple('Experience',
                        ['state', 'action_raw', 'log_prob', 'value', 'reward',
                         'throughput', 'agent_id', 'group_id', 'group_step',
                         'done', 'traj_id', 'step_in_traj'])
                ``throughput`` lets the learner compute the per-step fairness
                reward across simultaneous flows. An OFF-POLICY MARL method
                (replay-based, e.g. ma_dreamer) drops log_prob/value and the
                terminal bootstrap and keeps (state, action, reward, next_state,
                done, agent_id, group_id, …) instead — adjust both the worker
                push sites and the learner buffer to match.

────────────────────────────────────────────────────────────────────────────
REQUIRED FUNCTIONS
────────────────────────────────────────────────────────────────────────────
  normalize_state(info: dict) -> np.ndarray              (per-agent, as single-agent)
  assert_checkpoint_action_compatible(ckpt, source) -> None
      Raise if the checkpoint's action/state identity disagrees with this model.

────────────────────────────────────────────────────────────────────────────
REQUIRED POLICY API
────────────────────────────────────────────────────────────────────────────
  Policy(STATE_DIM, hidden, n_layers=..., n_heads=...) : nn.Module
      Parameter-shared across agents. Used two ways:

      • Decentralized execution (worker), with N=1 — its own observation:
          act(state_np, deterministic=False)
              -> (action_raw, mult, log_prob, value, act_mu, act_sig)
      • Centralized training (learner), over the joint set of simultaneous
          agents reconstructed from group_id+group_step:
          forward_joint(states)            # states: (B, N, STATE_DIM)
              -> (action distribution params, ..., value, ...)
        The worker also calls forward_joint on its last observation to emit a
        terminal bootstrap value (push_bootstrap) for GAE — omit for off-policy.

Size the joint model from max_agents = sweep.flows: the orchestrator surfaces
sweep.flows into the resolved config so the learner can build a model wide enough
for the largest episode, and passes n_agents to each worker via SAO_N_AGENTS.
"""

# ── Implement the contract above, then delete this guard. ─────────────────────
raise NotImplementedError(
    'template_multi_agent/model.py is a stub — copy this package, rename it, '
    'and implement model.py. See algorithms/ma_td3/model.py for a reference.'
)

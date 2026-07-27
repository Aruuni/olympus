"""The deployment `build_policy` contract, per algorithm.

Model modules bind their state and action plugins at import time, so two
algorithms with different states cannot be imported into one process. Every
check therefore runs in a subprocess and reports JSON.

The equivalence tests are the important ones: they rebuild the actor exactly
the way `algorithms/<name>/worker.py` does and require `build_policy` to
produce the same actions. That pins the deployment path to the training
convention instead of to an independent reading of it.
"""

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MODELS = REPO / "olympus" / "models"

# algorithm -> state plugin used to size a synthetic checkpoint
SYNTHETIC = {
    "td3": "default_orca",
    "mbpo_td3": "default_orca",
    "sac": "default_orca",
    "orca": "orca_repo",
    "ma_td3": "astraea",
    "dreamer_v3": "dreamer",
    "ma_dreamer": "tempest",
    "recurrent_ppo": "default_orca",
}


def _probe(function, *args):
    """Run `function(*args)` from this module in a clean subprocess."""
    code = (
        "import json,sys;"
        "from olympus.tests.test_deployment_policies import " + function + ";"
        "print(json.dumps(" + function + "(*sys.argv[1:])))"
    )
    env = dict(os.environ, PYTHONPATH=str(REPO))
    result = subprocess.run(
        [sys.executable, "-c", code, *[str(a) for a in args]],
        capture_output=True, text=True, env=env, cwd=str(REPO))
    if result.returncode != 0:
        raise AssertionError(
            f"probe {function}{args} failed:\n{result.stderr[-3000:]}")
    return json.loads(result.stdout.strip().splitlines()[-1])


# ── subprocess-side helpers ──────────────────────────────────────────────────

def _synthetic_checkpoint(algorithm, module):
    """A checkpoint payload with freshly initialized weights."""
    import torch

    if algorithm == "ma_td3":
        actor = module.Actor(module.STATE_DIM, 5, 256, 128, "batchnorm")
        return {"actor": actor.state_dict(), "history_len": 5,
                "actor_norm": "batchnorm"}
    if algorithm == "recurrent_ppo":
        net = module.RecurrentPPONet(module.STATE_DIM, 256)
        return {"model": net.state_dict()}
    if algorithm in ("dreamer_v3", "ma_dreamer"):
        world_class = (module.LocalWorldModel if algorithm == "ma_dreamer"
                       else module.WorldModel)
        world = world_class(
            module.STATE_DIM, module.ACTION_DIM, 256, 128, 256, 8, 8, 255)
        actor = module.Actor(256, world.latent_dim, 256)
        key = ("local_world_model" if algorithm == "ma_dreamer"
               else "world_model")
        return {key: world.state_dict(), "actor": actor.state_dict()}
    if algorithm == "mbpo_td3":
        actor = module.Actor(module.STATE_DIM, 128)
        return {"actor": actor.state_dict()}
    if algorithm == "orca":
        actor = module.Actor(module.STATE_DIM, 256, 256)
        return {"actor": actor.state_dict()}
    actor = module.Actor(module.STATE_DIM, 128, 64)
    return {"actor": actor.state_dict()}
    del torch


def contract(algorithm, state_name):
    """Build a synthetic policy and exercise the whole Policy contract."""
    import importlib

    import numpy as np
    import torch

    os.environ["SAO_STATE"] = state_name
    os.environ.setdefault("SAO_ACTION_NAME", "cwnd_multiplier")
    torch.manual_seed(0)
    module = importlib.import_module(f"olympus.algorithms.{algorithm}.model")
    ckpt = _synthetic_checkpoint(algorithm, module)

    policy = module.build_policy(ckpt, {}, {}, device="cpu",
                                 deterministic=True)
    dim = int(module.STATE_DIM)
    states = [np.full(dim, 0.25, dtype=np.float32),
              np.full(dim, 0.50, dtype=np.float32),
              np.full(dim, 0.75, dtype=np.float32)]

    # Dreamer samples its RSSM posterior (as the worker does), so the RNG is
    # re-seeded rather than assuming inference is deterministic: a reset policy
    # must return to its initial state, not merely be reproducible.
    torch.manual_seed(1234)
    first = [policy.act(s) for s in states]
    policy.reset()
    torch.manual_seed(1234)
    second = [policy.act(s) for s in states]

    wrong_shape = False
    try:
        policy.act(np.zeros(dim + 1, dtype=np.float32))
    except ValueError:
        wrong_shape = True

    return {
        "algorithm": policy.algorithm,
        "state_dim": policy.state_dim,
        "module_state_dim": dim,
        "actions": [float(a) for a in first],
        "after_reset": [float(a) for a in second],
        "finite": all(np.isfinite(a) for a in first),
        "bounded": all(-1.0 <= a <= 1.0 for a in first),
        "rejects_wrong_shape": wrong_shape,
        "has_reset": callable(getattr(policy, "reset", None)),
    }


def _worker_reference(algorithm, module, ckpt, agent, states):
    """Actions from actor construction copied out of the algorithm's worker."""
    import numpy as np
    import torch

    if algorithm == "ma_td3":
        history = int(ckpt.get("history_len") or module.HISTORY_LEN_DEFAULT)
        norm = str(ckpt.get("actor_norm") or "batchnorm")
        module.set_expected_history(history)
        module.set_expected_actor_norm(norm)
        actor = module.Actor(
            module.STATE_DIM, history,
            int(agent.get("hidden", 256)), int(agent.get("hidden2", 128)),
            norm)
        actor.load_state_dict(ckpt["actor"], strict=False)
        actor.eval()
        buffer = np.zeros(history * module.STATE_DIM, dtype=np.float32)
        out = []
        for state in states:
            buffer = np.concatenate([buffer[module.STATE_DIM:], state])
            action, _ = actor.act(buffer, noise_std=0.0)
            out.append(float(action))
        return out

    if algorithm == "orca":
        hidden = int(agent.get("hidden", 256))
        head = int(agent.get("head_hidden", hidden))
        hidden, head = module.actor_arch_from_checkpoint(ckpt, hidden, head)
        actor = module.Actor(module.STATE_DIM, hidden, head)
        actor.load_state_dict(ckpt["actor"])
        actor.eval()
        return [float(actor.act(s, noise_std=0.0)[0]) for s in states]

    if algorithm in ("td3", "mbpo_td3"):
        hidden = int(agent.get("hidden", 128))
        if algorithm == "td3":
            head = agent.get("head_hidden")
            head = int(head) if head not in (None, "") else None
            hidden, head = module.actor_arch_from_checkpoint(ckpt, hidden, head)
            actor = module.Actor(module.STATE_DIM, hidden, head)
        else:
            actor = module.Actor(module.STATE_DIM, hidden)
        actor.load_state_dict(ckpt.get("actor", ckpt.get("actor_state_dict")))
        actor.eval()
        hid, out = None, []
        for state in states:
            action, _, hid = actor.act(state, hid, noise_std=0.0)
            out.append(float(action))
        return out

    if algorithm in ("dreamer_v3", "ma_dreamer"):
        training = {}
        hidden, embed, h_dim = 256, 128, 256
        if algorithm == "ma_dreamer":
            world = module.LocalWorldModel(
                module.STATE_DIM, module.ACTION_DIM, hidden, embed, h_dim,
                8, 8, 255)
            world.load_state_dict(ckpt["local_world_model"], strict=False)
        else:
            world = module.WorldModel(
                module.STATE_DIM, module.ACTION_DIM, hidden, embed, h_dim,
                8, 8, 255)
            world.load_state_dict(ckpt["world_model"], strict=False)
        actor = module.Actor(
            h_dim, world.latent_dim, hidden,
            log_std_min=float(training.get("actor_log_std_min", -2.0)),
            log_std_max=float(training.get("actor_log_std_max", 0.0)))
        actor.load_state_dict(ckpt["actor"], strict=False)
        world.eval()
        actor.eval()
        h, z = world.rssm.initial(1, torch.device("cpu"))
        a_prev = torch.zeros(1, module.ACTION_DIM)
        out = []
        for state in states:
            with torch.no_grad():
                s_t = torch.from_numpy(state).unsqueeze(0)
                embedding = world.encoder(s_t)
                h, z, _, _ = world.rssm.step(h, z, a_prev, embedding)
                action, _, _, _ = actor.act(h, z, deterministic=True)
            a_prev = action
            out.append(float(action.item()))
        return out

    raise AssertionError(f"no worker reference for {algorithm}")


def equivalence(algorithm, config_path, checkpoint_path):
    """FlowPolicy actions vs the algorithm worker's own construction."""
    import numpy as np
    import torch
    import yaml

    from olympus.deployment.model import FlowPolicy, _selected_config

    with open(config_path) as handle:
        cfg = yaml.safe_load(handle) or {}

    torch.manual_seed(0)
    policy = FlowPolicy(cfg, checkpoint_path)
    dim = policy.state_dim
    rng = np.random.default_rng(7)
    states = [rng.random(dim).astype(np.float32) for _ in range(6)]
    deployed = [policy.infer(s) for s in states]

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    agent, _training = _selected_config(cfg, policy.algorithm)
    torch.manual_seed(0)
    reference = _worker_reference(
        policy.algorithm, policy.module, ckpt, agent, states)

    return {
        "algorithm": policy.algorithm,
        "state_dim": dim,
        "deployed": deployed,
        "reference": reference,
        "max_abs_diff": max(
            abs(a - b) for a, b in zip(deployed, reference)),
    }


# ── tests ────────────────────────────────────────────────────────────────────

class BuildPolicyContractTest(unittest.TestCase):
    """Every algorithm satisfies the Policy contract with fresh weights."""

    def test_all_algorithms_build_and_act(self):
        for algorithm, state_name in SYNTHETIC.items():
            with self.subTest(algorithm=algorithm):
                result = _probe("contract", algorithm, state_name)
                self.assertEqual(result["algorithm"], algorithm)
                self.assertEqual(result["state_dim"],
                                 result["module_state_dim"])
                self.assertTrue(result["finite"], result)
                self.assertTrue(result["bounded"], result)
                self.assertTrue(result["has_reset"])
                self.assertTrue(result["rejects_wrong_shape"],
                                "act() must reject a wrong-sized state")

    def test_reset_restores_initial_behaviour(self):
        """A reset policy repeats its first action sequence exactly."""
        for algorithm, state_name in SYNTHETIC.items():
            with self.subTest(algorithm=algorithm):
                result = _probe("contract", algorithm, state_name)
                for first, second in zip(result["actions"],
                                         result["after_reset"]):
                    self.assertAlmostEqual(
                        first, second, places=6,
                        msg=f"{algorithm} kept state across reset(): {result}")


class WorkerEquivalenceTest(unittest.TestCase):
    """build_policy reproduces each worker's own inference path."""

    CASES = {
        "ma_td3": (
            MODELS / "reps/ma_td3_20260723-234929",
            "checkpoints/ma_td3_450_cwnd_model.pt"),
        "orca": (
            MODELS / "reps/orca_20260720-213851",
            "checkpoints/orca_400_cwnd_model.pt"),
        "dreamer_v3": (
            MODELS / "olympus/dreamer_v3_20260724-233950",
            "checkpoints/dreamer_v3_cwnd_model.pt"),
        "ma_dreamer": (
            MODELS / "ma_dreamer_20260612-180302",
            "checkpoints/ma_dreamer_cwnd_model.pt"),
    }

    def test_matches_worker_construction(self):
        for algorithm, (run_dir, relative) in self.CASES.items():
            config = run_dir / "telemetry" / "config.resolved.yaml"
            checkpoint = run_dir / relative
            with self.subTest(algorithm=algorithm):
                if not (config.is_file() and checkpoint.is_file()):
                    self.skipTest(f"no local checkpoint for {algorithm}")
                result = _probe("equivalence", algorithm, config, checkpoint)
                self.assertEqual(result["algorithm"], algorithm)
                self.assertLess(
                    result["max_abs_diff"], 1e-6,
                    f"{algorithm}: deployment diverges from worker "
                    f"{result['deployed']} vs {result['reference']}")


if __name__ == "__main__":
    unittest.main()

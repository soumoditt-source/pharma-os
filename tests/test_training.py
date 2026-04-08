from __future__ import annotations

import numpy as np

from train_ppo import CURRICULUM_TASKS, OBSERVATION_DIM, PharmaGymWrapper, evaluate_policy


def test_training_wrapper_uses_real_environment():
    env = PharmaGymWrapper(task="qed_optimizer")

    observation, info = env.reset(seed=202)
    assert observation.shape == (OBSERVATION_DIM,)
    assert np.isfinite(observation).all()
    assert info["task"] == "qed_optimizer"

    next_observation, reward, terminated, truncated, step_info = env.step(0)
    assert next_observation.shape == (OBSERVATION_DIM,)
    assert isinstance(float(reward), float)
    assert isinstance(terminated, bool)
    assert truncated is False
    assert "chosen_smiles" in step_info


class _GreedyStubModel:
    def predict(self, observation, deterministic=True):
        return 0, None


def test_evaluate_policy_returns_research_metrics():
    metrics = evaluate_policy(_GreedyStubModel(), task="lipinski_optimizer", episodes=1, seed=101)

    assert metrics["episodes"] == 1.0
    assert 0.0 <= metrics["success_rate"] <= 1.0
    assert metrics["mean_unique_molecules"] >= 1.0
    assert "multi_objective_designer" in CURRICULUM_TASKS

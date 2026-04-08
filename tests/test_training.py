from __future__ import annotations

import numpy as np

from train_ppo import OBSERVATION_DIM, PharmaGymWrapper


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

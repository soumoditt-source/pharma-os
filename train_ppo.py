from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import gymnasium as gym
import numpy as np

from models import PharmaAction
from server.environment import (
    IMPROVED_MOLECULE_HINTS,
    LIPINSKI_START_MOLECULES,
    MULTI_OBJ_START_MOLECULES,
    QED_START_MOLECULES,
    PharmaEnvironment,
)
from models import TASK_SUCCESS_THRESHOLDS


def _dedupe(values: List[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


TASK_CANDIDATE_LIBRARY: Dict[str, List[str]] = {
    "lipinski_optimizer": _dedupe(
        IMPROVED_MOLECULE_HINTS["lipinski_optimizer"] + LIPINSKI_START_MOLECULES
    ),
    "qed_optimizer": _dedupe(
        IMPROVED_MOLECULE_HINTS["qed_optimizer"] + QED_START_MOLECULES
    ),
    "multi_objective_designer": _dedupe(
        IMPROVED_MOLECULE_HINTS["multi_objective_designer"] + MULTI_OBJ_START_MOLECULES
    ),
}
CURRICULUM_TASKS = [
    "lipinski_optimizer",
    "qed_optimizer",
    "multi_objective_designer",
]

OBSERVATION_DIM = 19


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


class PharmaGymWrapper(gym.Env):
    """
    Gymnasium wrapper around the real PharmaOS environment.

    Actions are drawn from a curated medicinal-chemistry proposal catalog and
    rewards come directly from the real environment, making this a truthful PPO
    baseline instead of a random mock.
    """

    metadata = {"render_modes": []}

    def __init__(self, task: str = "qed_optimizer"):
        super().__init__()
        self.task = task
        self.env = PharmaEnvironment(task_name=task)
        self.catalog = TASK_CANDIDATE_LIBRARY[task]
        self.action_space = gym.spaces.Discrete(len(self.catalog))
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(OBSERVATION_DIM,),
            dtype=np.float32,
        )

    def _encode_observation(self, observation) -> np.ndarray:
        props = observation.properties
        admet = observation.admet

        sa_norm = (
            _clip01((10.0 - float(props.sa_score or 10.0)) / 9.0)
            if props.sa_score is not None
            else 0.0
        )
        log_s = float(props.logS or -8.0)
        log_s_norm = _clip01((log_s + 8.0) / 10.0)

        features = np.array(
            [
                _clip01((props.molecular_weight or 0.0) / 600.0),
                _clip01(((props.logp or 0.0) + 2.0) / 8.0),
                _clip01((props.hbd or 0) / 10.0),
                _clip01((props.hba or 0) / 15.0),
                _clip01((props.tpsa or 0.0) / 200.0),
                _clip01((props.rotatable_bonds or 0) / 15.0),
                _clip01(props.qed or 0.0),
                sa_norm,
                _clip01(props.fsp3 or 0.0),
                _clip01(1.0 - ((props.lipinski_violations or 4) / 4.0)),
                _clip01(props.fingerprint_similarity or 0.0),
                log_s_norm,
                _clip01(props.bbb_score or 0.0),
                _clip01(1.0 - float(props.herg_risk or 1.0)),
                1.0 if bool(props.pains_alert) else 0.0,
                _clip01(observation.step_count / max(self.env.get_state().max_steps, 1)),
                _clip01(observation.best_score),
                _clip01(observation.properties.composite_score or observation.best_score or 0.0),
                _clip01(observation.visited_count / max(len(self.catalog), 1)),
            ],
            dtype=np.float32,
        )
        if admet is not None:
            features[12] = _clip01(admet.bbb_score or features[12])
            features[13] = _clip01(1.0 - float(admet.herg_risk or (1.0 - features[13])))
        return features

    def reset(self, seed: int | None = None, options: Dict[str, Any] | None = None):
        super().reset(seed=seed)
        observation = self.env.reset(seed=seed)
        return self._encode_observation(observation), {
            "task": self.task,
            "current_smiles": observation.current_smiles,
        }

    def step(self, action: int):
        chosen_smiles = self.catalog[int(action)]
        observation, reward, done, info = self.env.step(
            PharmaAction(smiles=chosen_smiles, reasoning="ppo_catalog_action")
        )
        info = {
            **info,
            "chosen_smiles": chosen_smiles,
            "best_score": observation.best_score,
        }
        return self._encode_observation(observation), float(reward), bool(done), False, info


def train_agent(
    task: str,
    timesteps: int,
    seed: int,
    output_dir: Path,
    device: str,
) -> Path:
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv

    def make_env():
        return Monitor(PharmaGymWrapper(task=task))

    env = DummyVecEnv([make_env])
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        seed=seed,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=64,
        gamma=0.99,
        device=device,
        tensorboard_log=str(output_dir / "tensorboard"),
    )
    model.learn(total_timesteps=timesteps, progress_bar=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / f"ppo_pharmaos_{task}"
    model.save(str(model_path))

    metadata = {
        "task": task,
        "timesteps": timesteps,
        "seed": seed,
        "device": device,
        "catalog_size": len(TASK_CANDIDATE_LIBRARY[task]),
        "model_path": str(model_path.with_suffix(".zip")),
    }
    (output_dir / f"ppo_pharmaos_{task}.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    return model_path.with_suffix(".zip")


def evaluate_policy(model: Any, task: str, episodes: int = 5, seed: int = 11) -> Dict[str, float]:
    final_scores: List[float] = []
    rewards: List[float] = []
    successes = 0
    unique_molecules: List[int] = []
    unique_scaffolds: List[int] = []
    steps_taken: List[int] = []

    env = PharmaGymWrapper(task=task)
    for episode_idx in range(episodes):
        observation, _ = env.reset(seed=seed + episode_idx)
        done = False
        cumulative_reward = 0.0
        steps = 0
        while not done:
            action, _ = model.predict(observation, deterministic=True)
            action_value = int(action.item()) if hasattr(action, "item") else int(action)
            observation, reward, terminated, truncated, _ = env.step(action_value)
            cumulative_reward += float(reward)
            done = bool(terminated or truncated)
            steps += 1

        state = env.env.get_state()
        final_scores.append(float(state.best_score))
        rewards.append(cumulative_reward)
        unique_molecules.append(len(state.visited_molecules))
        unique_scaffolds.append(state.unique_scaffolds)
        steps_taken.append(steps)
        if state.best_score >= TASK_SUCCESS_THRESHOLDS[task]:
            successes += 1

    return {
        "episodes": float(episodes),
        "mean_final_score": float(np.mean(final_scores)),
        "mean_reward": float(np.mean(rewards)),
        "success_rate": float(successes / max(episodes, 1)),
        "mean_unique_molecules": float(np.mean(unique_molecules)),
        "mean_unique_scaffolds": float(np.mean(unique_scaffolds)),
        "mean_steps": float(np.mean(steps_taken)),
    }


def train_curriculum(
    total_timesteps: int,
    seed: int,
    output_dir: Path,
    device: str,
    eval_episodes: int = 3,
) -> Path:
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv

    stage_timesteps = max(1, total_timesteps // len(CURRICULUM_TASKS))

    def make_env(task: str):
        return DummyVecEnv([lambda: Monitor(PharmaGymWrapper(task=task))])

    model = PPO(
        "MlpPolicy",
        make_env(CURRICULUM_TASKS[0]),
        verbose=1,
        seed=seed,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=64,
        gamma=0.99,
        device=device,
        tensorboard_log=str(output_dir / "tensorboard"),
    )

    stage_summaries: List[Dict[str, Any]] = []
    for stage_index, task in enumerate(CURRICULUM_TASKS):
        model.set_env(make_env(task))
        model.learn(total_timesteps=stage_timesteps, progress_bar=True, reset_num_timesteps=(stage_index == 0))
        metrics = evaluate_policy(model, task=task, episodes=eval_episodes, seed=seed + 100 * (stage_index + 1))
        stage_summaries.append(
            {
                "task": task,
                "timesteps": stage_timesteps,
                "metrics": metrics,
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "ppo_pharmaos_curriculum"
    model.save(str(model_path))
    (output_dir / "ppo_pharmaos_curriculum.json").write_text(
        json.dumps(
            {
                "mode": "curriculum",
                "seed": seed,
                "device": device,
                "stages": stage_summaries,
                "model_path": str(model_path.with_suffix(".zip")),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return model_path.with_suffix(".zip")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a PPO baseline on the real PharmaOS environment.")
    parser.add_argument(
        "--task",
        choices=sorted(TASK_CANDIDATE_LIBRARY.keys()),
        default="qed_optimizer",
        help="Task to train on.",
    )
    parser.add_argument("--timesteps", type=int, default=5000, help="Total PPO timesteps.")
    parser.add_argument("--seed", type=int, default=7, help="Training seed.")
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device to use. Examples: auto, cpu, cuda.",
    )
    parser.add_argument(
        "--output-dir",
        default="trained_agents",
        help="Directory for trained model artifacts.",
    )
    parser.add_argument(
        "--curriculum",
        action="store_true",
        help="Train sequentially across easy, medium, and hard tasks.",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=3,
        help="Evaluation episodes per curriculum stage.",
    )
    args = parser.parse_args()

    if args.curriculum:
        model_path = train_curriculum(
            total_timesteps=args.timesteps,
            seed=args.seed,
            output_dir=Path(args.output_dir),
            device=args.device,
            eval_episodes=args.eval_episodes,
        )
    else:
        model_path = train_agent(
            task=args.task,
            timesteps=args.timesteps,
            seed=args.seed,
            output_dir=Path(args.output_dir),
            device=args.device,
        )
    print(f"Saved PPO baseline to {model_path}")


if __name__ == "__main__":
    main()

"""
PharmaOS — Drug Discovery Molecular Optimization RL Environment
client.py: HTTPEnvClient for type-safe communication with the server.

Usage:
    from client import PharmaEnvClient, PharmaAction

    # Sync (using .sync() wrapper):
    with PharmaEnvClient(base_url="http://localhost:8000").sync() as client:
        result = client.reset()
        result = client.step(PharmaAction(smiles="CC(=O)Nc1ccc(cc1)O"))

    # Async:
    async with PharmaEnvClient(base_url="http://localhost:8000") as client:
        result = await client.reset()

Built by: Team Fullstack Shinobi & Soumoditya Das
Event: Meta x PyTorch OpenEnv Hackathon 2026
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any, Dict, Optional

import requests

from models import (
    PharmaAction,
    PharmaObservation,
    PharmaState,
    AVAILABLE_TASKS,
)


class StepResult:
    """Holds result from reset() or step()."""

    def __init__(self, observation: PharmaObservation, reward: Optional[float], done: bool):
        self.observation = observation
        self.reward = reward
        self.done = done

    def __repr__(self) -> str:
        reward_text = "None" if self.reward is None else f"{self.reward:.4f}"
        return (
            f"StepResult(done={self.done}, reward={reward_text}, "
            f"score={self.observation.best_score:.4f}, "
            f"smiles={self.observation.current_smiles[:40]})"
        )


class PharmaEnvClient:
    """
    HTTP client for the PharmaOS environment.
    Implements the OpenEnv interface: reset(), step(), state().
    """

    def __init__(self, base_url: str = "http://localhost:8000", task: str = "lipinski_optimizer"):
        self.base_url = base_url.rstrip("/")
        self.task = task
        self._session = requests.Session()

    # -----------------------------------------------------------------------
    # Core interface (sync)
    # -----------------------------------------------------------------------

    def reset(self, task: Optional[str] = None, seed: Optional[int] = None) -> StepResult:
        """Start a new episode. Returns initial StepResult."""
        task_name = task or self.task
        payload: Dict[str, Any] = {"task": task_name}
        if seed is not None:
            payload["seed"] = seed
        resp = self._session.post(
            f"{self.base_url}/reset",
            json=payload,
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        obs = PharmaObservation(**data.get("observation", data))
        return StepResult(observation=obs, reward=None, done=False)

    def step(self, action: PharmaAction) -> StepResult:
        """Execute one step. Returns StepResult."""
        resp = self._session.post(
            f"{self.base_url}/step",
            json=action.model_dump(),
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        reward = data.pop("reward", 0.0)
        done = data.pop("done", False)
        data.pop("info", None)
        obs = PharmaObservation(**data.get("observation", data))
        return StepResult(observation=obs, reward=reward, done=done)

    def state(self) -> PharmaState:
        """Get current episode state metadata."""
        resp = self._session.get(f"{self.base_url}/state", timeout=30)
        resp.raise_for_status()
        data = resp.json()
        return PharmaState(**data.get("state", data))

    def health(self) -> Dict[str, str]:
        """Check server health."""
        resp = self._session.get(f"{self.base_url}/health", timeout=10)
        resp.raise_for_status()
        return resp.json()

    def close(self):
        self._session.close()

    # Context manager support
    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def sync(self):
        """Return self for sync usage (compatibility alias)."""
        return self

    # -----------------------------------------------------------------------
    # Factory methods
    # -----------------------------------------------------------------------

    @classmethod
    def from_env(cls, task: str = "lipinski_optimizer") -> "PharmaEnvClient":
        """Create client using PHARMAO_URL environment variable."""
        url = os.getenv("PHARMAO_URL", "http://localhost:8000")
        return cls(base_url=url, task=task)

    @classmethod
    def from_hf_space(cls, space_id: str, task: str = "lipinski_optimizer") -> "PharmaEnvClient":
        """
        Create client pointing to a Hugging Face Space.
        space_id: e.g. "soumoditya/pharma-os"
        """
        owner, name = space_id.split("/")
        url = f"https://{owner}-{name}.hf.space"
        return cls(base_url=url, task=task)

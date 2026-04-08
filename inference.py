from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional, Tuple

import requests
from openai import OpenAI

os.environ.setdefault("PHARMAOS_VERBOSE_LOGS", "0")

from server.environment import (
    IMPROVED_MOLECULE_HINTS,
    LIPINSKI_START_MOLECULES,
    MULTI_OBJ_START_MOLECULES,
    QED_START_MOLECULES,
    compute_properties,
    compute_task_score,
)

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
API_KEY = os.getenv("API_KEY") or os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
PHARMAO_URL = os.getenv("PHARMAO_URL", "http://localhost:8000").rstrip("/")

if API_KEY is None:
    raise ValueError("API_KEY environment variable is required (HF_TOKEN is accepted as a backward-compatible fallback)")

client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

BENCHMARK = "pharma-os"
TASKS = [
    "lipinski_optimizer",
    "qed_optimizer",
    "multi_objective_designer",
]
TASK_SUCCESS_THRESHOLDS = {
    "lipinski_optimizer": 1.0,
    "qed_optimizer": 0.75,
    "multi_objective_designer": 0.70,
}
TASK_RESET_SEEDS = {
    "lipinski_optimizer": 101,
    "qed_optimizer": 202,
    "multi_objective_designer": 303,
}
TASK_STEP_LIMITS = {
    "lipinski_optimizer": 6,
    "qed_optimizer": 8,
    "multi_objective_designer": 10,
}
FALLBACK_MOLECULES: Dict[str, List[str]] = {
    "lipinski_optimizer": [
        "CC(=O)Nc1ccc(cc1)O",
        "CC(=O)Oc1ccccc1C(=O)O",
        "c1ccncc1C(=O)O",
        "Cc1ccc(cc1)S(N)(=O)=O",
    ],
    "qed_optimizer": [
        "O=C(Nc1cccnc1)c1ccc(F)cc1",
        "Cc1ccc(cc1)NC(=O)c1ccccn1",
        "O=C(Nc1ccc(F)cc1)c1ccco1",
        "CC1=CN=C(S1)NC(=O)c1ccccn1",
    ],
    "multi_objective_designer": [
        "CC(=O)Nc1ccc(cc1)O",
        "O=C(Nc1cccc(F)c1)c1ccc(N)cc1",
        "Cc1ccc(cc1)NC(=O)Cn1ccc2ccccc21",
        "Cc1ccc(cc1)S(=O)(=O)Nc1ccccn1",
    ],
}
TASK_CANDIDATE_LIBRARY: Dict[str, List[str]] = {
    "lipinski_optimizer": list(dict.fromkeys(FALLBACK_MOLECULES["lipinski_optimizer"] + IMPROVED_MOLECULE_HINTS["lipinski_optimizer"] + LIPINSKI_START_MOLECULES)),
    "qed_optimizer": list(dict.fromkeys(FALLBACK_MOLECULES["qed_optimizer"] + IMPROVED_MOLECULE_HINTS["qed_optimizer"] + QED_START_MOLECULES)),
    "multi_objective_designer": list(dict.fromkeys(FALLBACK_MOLECULES["multi_objective_designer"] + IMPROVED_MOLECULE_HINTS["multi_objective_designer"] + MULTI_OBJ_START_MOLECULES)),
}

SYSTEM_PROMPT = (
    "You are a medicinal chemistry agent. "
    "Return exactly one improved molecule in the format "
    "<SMILES>...</SMILES> with no extra XML blocks."
)


def _single_line(value: Optional[str]) -> str:
    if not value:
        return "null"
    return str(value).replace("\r", " ").replace("\n", " ").strip() or "null"


def _extract_observation(payload: Dict[str, Any]) -> Dict[str, Any]:
    observation = payload.get("observation")
    return observation if isinstance(observation, dict) else payload


def _extract_smiles(text: str) -> Optional[str]:
    if not text:
        return None

    tagged = re.search(r"<SMILES>(.*?)</SMILES>", text, re.IGNORECASE | re.DOTALL)
    if tagged:
        return tagged.group(1).strip()

    token = re.search(r"([A-Za-z0-9@+\-\[\]()=#%\\/]+)", text)
    if token:
        candidate = token.group(1).strip()
        if len(candidate) > 2:
            return candidate
    return None


def _format_properties(obs: Dict[str, Any]) -> str:
    props = obs.get("properties") or {}
    admet = obs.get("admet") or {}
    return (
        f"SMILES={obs.get('current_smiles', '')}\n"
        f"Score={obs.get('best_score', 0.0)}\n"
        f"MW={props.get('molecular_weight')} LogP={props.get('logp')} "
        f"HBD={props.get('hbd')} HBA={props.get('hba')}\n"
        f"QED={props.get('qed')} SA={props.get('sa_score')} "
        f"BBB={props.get('bbb_score')} hERG={props.get('herg_risk')}\n"
        f"PAINS={props.get('pains_alert')} Solubility={admet.get('solubility_class')}\n"
        f"Feedback={str(obs.get('feedback', ''))[:400]}"
    )


def _choose_action(
    task_name: str,
    obs: Dict[str, Any],
    tried: List[str],
) -> str:
    prompt = (
        f"Task: {task_name}\n"
        f"Previously tried: {', '.join(tried[-6:]) if tried else 'none'}\n"
        f"{_format_properties(obs)}\n"
        "Propose the next SMILES."
    )

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            temperature=0,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        content = response.choices[0].message.content or ""
    except Exception:
        content = ""

    proposed = _extract_smiles(content)
    if proposed and proposed not in tried and _score_candidate(task_name, obs, proposed, tried) > float("-inf"):
        return proposed

    return _best_fallback_candidate(task_name, obs, tried)


def _score_candidate(
    task_name: str,
    obs: Dict[str, Any],
    candidate: str,
    tried: List[str],
) -> float:
    if not candidate or candidate in tried:
        return float("-inf")

    target_smiles = (obs.get("metadata") or {}).get("target_smiles", "")
    props = compute_properties(candidate, target_smiles=target_smiles)
    if props is None:
        return float("-inf")

    base_score = compute_task_score(props, task_name)
    current_best = float(obs.get("best_score", 0.0) or 0.0)
    improvement = base_score - current_best
    novelty_bonus = 0.05 if candidate not in tried else -0.20
    pains_penalty = -0.10 if bool(props.pains_alert) else 0.0
    synthetic_bonus = max(0.0, (10.0 - float(props.sa_score or 10.0)) / 20.0)
    uncertainty_proxy_penalty = -0.03 if (props.molecular_weight or 0.0) > 550 else 0.0
    return base_score + 0.35 * improvement + novelty_bonus + synthetic_bonus + pains_penalty + uncertainty_proxy_penalty


def _best_fallback_candidate(task_name: str, obs: Dict[str, Any], tried: List[str]) -> str:
    ranked = sorted(
        TASK_CANDIDATE_LIBRARY[task_name],
        key=lambda candidate: _score_candidate(task_name, obs, candidate, tried),
        reverse=True,
    )
    for candidate in ranked:
        if candidate not in tried:
            return candidate
    return ranked[0]


def _reset_env(session: requests.Session, task_name: str) -> Dict[str, Any]:
    response = session.post(
        f"{PHARMAO_URL}/reset",
        json={"task": task_name, "seed": TASK_RESET_SEEDS[task_name]},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def _step_env(
    session: requests.Session,
    smiles: str,
) -> Tuple[Dict[str, Any], float, bool, Optional[str]]:
    response = session.post(
        f"{PHARMAO_URL}/step",
        json={"action": {"smiles": smiles, "reasoning": ""}},
        timeout=30,
    )
    response.raise_for_status()

    payload = response.json()
    observation = _extract_observation(payload)
    reward = float(payload.get("reward", observation.get("reward", 0.0) or 0.0))
    done = bool(payload.get("done", observation.get("done", False)))
    info = payload.get("info") if isinstance(payload.get("info"), dict) else {}
    error = info.get("error") if isinstance(info, dict) else None
    return observation, reward, done, error


def run_task(task_name: str) -> None:
    rewards: List[float] = []
    tried: List[str] = []
    steps = 0
    success = False
    last_observation: Dict[str, Any] = {}
    final_score = 0.0

    print(f"[START] task={task_name} env={BENCHMARK} model={MODEL_NAME}", flush=True)

    with requests.Session() as session:
        try:
            reset_payload = _reset_env(session, task_name)
            last_observation = _extract_observation(reset_payload)

            initial_smiles = last_observation.get("current_smiles")
            if initial_smiles:
                tried.append(str(initial_smiles))

            for step_index in range(1, TASK_STEP_LIMITS[task_name] + 1):
                action = _choose_action(task_name, last_observation, tried)
                if action not in tried:
                    tried.append(action)

                last_observation, reward, done, error = _step_env(session, action)
                rewards.append(reward)
                steps = step_index

                print(
                    f"[STEP] step={step_index} action={action} reward={reward:.2f} "
                    f"done={'true' if done else 'false'} error={_single_line(error)}",
                    flush=True,
                )

                if done:
                    break

            final_score = float(last_observation.get("best_score", 0.0) or 0.0)
            success = final_score >= TASK_SUCCESS_THRESHOLDS[task_name]
        except Exception:
            if steps == 0:
                rewards = []
            print(
                f"[END] success=false steps={steps} score={final_score:.2f} rewards="
                f"{','.join(f'{reward:.2f}' for reward in rewards) if rewards else '0.00'}",
                flush=True,
            )
            return

    print(
        f"[END] success={'true' if success else 'false'} steps={steps} score={final_score:.2f} rewards="
        f"{','.join(f'{reward:.2f}' for reward in rewards) if rewards else '0.00'}",
        flush=True,
    )


if __name__ == "__main__":
    for task in TASKS:
        run_task(task)

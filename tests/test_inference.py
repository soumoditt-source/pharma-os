from __future__ import annotations

import importlib
import io
import re
import sys
from contextlib import redirect_stdout


def _load_inference(monkeypatch):
    monkeypatch.setenv("API_KEY", "test-token")
    monkeypatch.setenv("API_BASE_URL", "http://127.0.0.1:9/v1")
    monkeypatch.setenv("MODEL_NAME", "test-model")
    monkeypatch.setenv("PHARMAO_URL", "http://127.0.0.1:8000")
    sys.modules.pop("inference", None)
    return importlib.import_module("inference")


def test_inference_accepts_hf_token_fallback(monkeypatch):
    monkeypatch.delenv("API_KEY", raising=False)
    monkeypatch.setenv("HF_TOKEN", "legacy-token")
    monkeypatch.setenv("API_BASE_URL", "http://127.0.0.1:9/v1")
    monkeypatch.setenv("MODEL_NAME", "test-model")
    monkeypatch.setenv("PHARMAO_URL", "http://127.0.0.1:8000")
    sys.modules.pop("inference", None)

    inference = importlib.import_module("inference")

    assert inference.API_KEY == "legacy-token"


def test_warm_litellm_proxy_attempts_one_client_call(monkeypatch):
    inference = _load_inference(monkeypatch)
    calls = []

    def _fake_create(**kwargs):
        calls.append(kwargs)
        raise RuntimeError("proxy unavailable")

    monkeypatch.setattr(inference.client.chat.completions, "create", _fake_create)
    inference._PROXY_WARMED = False

    inference._warm_litellm_proxy()
    inference._warm_litellm_proxy()

    assert len(calls) == 1
    assert calls[0]["model"] == "test-model"


def test_run_task_emits_score_in_end_line(monkeypatch):
    inference = _load_inference(monkeypatch)

    monkeypatch.setattr(
        inference,
        "_reset_env",
        lambda session, task_name: {"observation": {"current_smiles": "c1ccccc1", "best_score": 0.10}},
    )
    monkeypatch.setattr(inference, "_choose_action", lambda task_name, obs, tried: "CCO")
    monkeypatch.setattr(
        inference,
        "_step_env",
        lambda session, smiles: (
            {"current_smiles": smiles, "best_score": 1.00},
            0.25,
            True,
            None,
        ),
    )

    stdout = io.StringIO()
    with redirect_stdout(stdout):
        inference.run_task("lipinski_optimizer")

    lines = [line.strip() for line in stdout.getvalue().splitlines() if line.strip()]
    assert lines[0].startswith("[START] task=lipinski_optimizer env=pharma-os model=test-model")
    assert lines[1] == "[STEP] step=1 action=CCO reward=0.25 done=true error=null"
    assert lines[2] == "[END] success=true steps=1 score=0.99 rewards=0.25"


def test_run_task_keeps_stdout_strict_when_candidate_scoring_runs(monkeypatch):
    inference = _load_inference(monkeypatch)

    monkeypatch.setattr(
        inference,
        "_reset_env",
        lambda session, task_name: {
            "observation": {
                "current_smiles": "CCCCCCCC(=O)OCCO",
                "best_score": 0.10,
                "metadata": {},
            }
        },
    )
    monkeypatch.setattr(
        inference,
        "_step_env",
        lambda session, smiles: (
            {"current_smiles": smiles, "best_score": 0.80},
            0.30,
            True,
            None,
        ),
    )

    stdout = io.StringIO()
    with redirect_stdout(stdout):
        inference.run_task("qed_optimizer")

    lines = [line.strip() for line in stdout.getvalue().splitlines() if line.strip()]
    assert len(lines) == 3
    assert re.fullmatch(r"\[START\] task=qed_optimizer env=pharma-os model=test-model", lines[0])
    assert re.fullmatch(
        r"\[STEP\] step=1 action=.* reward=0\.30 done=true error=null",
        lines[1],
    )
    assert lines[2] == "[END] success=true steps=1 score=0.80 rewards=0.30"

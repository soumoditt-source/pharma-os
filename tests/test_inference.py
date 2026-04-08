from __future__ import annotations

import importlib
import io
import sys
from contextlib import redirect_stdout


def _load_inference(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "test-token")
    monkeypatch.setenv("API_BASE_URL", "http://127.0.0.1:9/v1")
    monkeypatch.setenv("MODEL_NAME", "test-model")
    monkeypatch.setenv("PHARMAO_URL", "http://127.0.0.1:8000")
    sys.modules.pop("inference", None)
    return importlib.import_module("inference")


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
    assert lines[2] == "[END] success=true steps=1 score=1.00 rewards=0.25"

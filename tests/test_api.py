from __future__ import annotations

from fastapi.testclient import TestClient

import server.app as app_module
from server.app import app


def test_reset_returns_legacy_and_openenv_shapes():
    client = TestClient(app)

    response = client.post("/reset", json={"task": "lipinski_optimizer"})
    assert response.status_code == 200

    payload = response.json()
    assert "observation" in payload
    assert payload["observation"]["task_name"] == "lipinski_optimizer"
    assert payload["current_smiles"] == payload["observation"]["current_smiles"]
    assert payload["done"] is False


def test_step_accepts_openenv_action_envelope_after_reset():
    client = TestClient(app)

    reset_response = client.post("/reset", json={"task": "qed_optimizer"})
    assert reset_response.status_code == 200

    step_response = client.post(
        "/step",
        json={"action": {"smiles": "O=C(Nc1cccnc1)c1ccc(F)cc1", "reasoning": ""}},
    )
    assert step_response.status_code == 200

    payload = step_response.json()
    assert "observation" in payload
    assert payload["observation"]["task_name"] == "qed_optimizer"
    assert isinstance(payload["reward"], float)
    assert isinstance(payload["done"], bool)


def test_runtime_compatibility_endpoints_exist():
    client = TestClient(app)

    metadata_response = client.get("/metadata")
    schema_response = client.get("/schema")
    runtime_response = client.get("/api/runtime_status")
    mcp_response = client.post("/mcp", json={})

    assert metadata_response.status_code == 200
    assert metadata_response.json()["name"] == "pharma-os"

    schema_payload = schema_response.json()
    assert schema_response.status_code == 200
    assert {"action", "observation", "state"} <= set(schema_payload.keys())

    runtime_payload = runtime_response.json()
    assert runtime_response.status_code == 200
    assert {"rdkit_available", "draw_available", "llm_provider_mode", "llm_backends"} <= set(runtime_payload.keys())

    mcp_payload = mcp_response.json()
    assert mcp_response.status_code == 200
    assert mcp_payload["jsonrpc"] == "2.0"


def test_state_before_reset_keeps_best_score_in_open_interval():
    app_module._http_env = None
    client = TestClient(app)

    response = client.get("/state")
    assert response.status_code == 200

    payload = response.json()
    assert 0.0 < payload["best_score"] < 1.0
    assert 0.0 < payload["state"]["best_score"] < 1.0


def test_schema_best_score_defaults_stay_in_open_interval():
    client = TestClient(app)

    response = client.get("/schema")
    assert response.status_code == 200

    payload = response.json()
    observation_default = payload["observation"]["properties"]["best_score"]["default"]
    state_default = payload["state"]["properties"]["best_score"]["default"]

    assert 0.0 < observation_default < 1.0
    assert 0.0 < state_default < 1.0


def test_reset_accepts_seed_and_is_reproducible():
    client = TestClient(app)

    first = client.post("/reset", json={"task": "multi_objective_designer", "seed": 303})
    second = client.post("/reset", json={"task": "multi_objective_designer", "seed": 303})

    assert first.status_code == 200
    assert second.status_code == 200

    first_obs = first.json()["observation"]
    second_obs = second.json()["observation"]

    assert first_obs["current_smiles"] == second_obs["current_smiles"]
    assert first_obs["metadata"]["episode_seed"] == 303
    assert second_obs["metadata"]["episode_seed"] == 303


def test_structure_endpoint_returns_renderable_payload():
    client = TestClient(app)

    response = client.get("/api/structure", params={"smiles": "CC(=O)Nc1ccc(cc1)O"})
    assert response.status_code == 200

    payload = response.json()
    assert payload["smiles"]
    assert payload["structure_source"] in {"rdkit", "unavailable"}
    if payload["structure_source"] == "rdkit":
        assert payload["molblock_2d"] is not None

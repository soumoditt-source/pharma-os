from __future__ import annotations

from fastapi.testclient import TestClient

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
    mcp_response = client.post("/mcp", json={})

    assert metadata_response.status_code == 200
    assert metadata_response.json()["name"] == "pharma-os"

    schema_payload = schema_response.json()
    assert schema_response.status_code == 200
    assert {"action", "observation", "state"} <= set(schema_payload.keys())

    mcp_payload = mcp_response.json()
    assert mcp_response.status_code == 200
    assert mcp_payload["jsonrpc"] == "2.0"


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

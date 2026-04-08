from __future__ import annotations

from types import SimpleNamespace

from server.agent import PharmaAgent


def test_agent_uses_validated_compound_lookup_for_common_query(monkeypatch):
    monkeypatch.setattr(
        "server.agent.resolve_compound_query",
        lambda query: {
            "summary": "Detergent is a product class.",
            "results": [
                {
                    "name": "Sodium Lauryl Sulfate",
                    "loadable": True,
                },
                {
                    "name": "Sodium Laureth Sulfate",
                    "loadable": True,
                },
            ],
        },
    )

    agent = PharmaAgent()
    trace = agent.get_reasoning_trace("detergent")

    assert trace.formula == "Validated Compound Resolver"
    assert "Sodium Lauryl Sulfate" in trace.recommendation


def test_agent_can_fail_over_from_hf_to_mistral(monkeypatch):
    created_clients = []

    class _FakeClient:
        def __init__(self, base_url=None, api_key=None):
            self.base_url = base_url
            self.api_key = api_key
            created_clients.append(self)
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

        def _create(self, model, **kwargs):
            if "huggingface" in self.base_url:
                raise RuntimeError("primary unavailable")
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content=f"Mistral guidance via {model}"
                        )
                    )
                ]
            )

    monkeypatch.setenv("LLM_PROVIDER", "auto")
    monkeypatch.setenv("HF_TOKEN", "hf-token")
    monkeypatch.setenv("API_BASE_URL", "https://router.huggingface.co/v1")
    monkeypatch.setenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
    monkeypatch.setenv("MISTRAL_API_KEY", "mistral-token")
    monkeypatch.setenv("MISTRAL_API_BASE_URL", "https://api.mistral.ai/v1")
    monkeypatch.setenv("MISTRAL_MODEL_NAME", "mistral-small-latest")
    monkeypatch.setattr("server.agent.OpenAI", _FakeClient)

    agent = PharmaAgent()
    trace = agent.get_reasoning_trace("Reduce hERG risk for this molecule.", context="Current molecule is lipophilic.")

    assert len(created_clients) == 2
    assert trace.level == "LLM_ORACLE"
    assert trace.provider == "mistral"
    assert "Mistral guidance" in trace.recommendation

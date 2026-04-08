from __future__ import annotations

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

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_project_log_does_not_contain_embedded_tokens():
    project_log = (ROOT / "PROJECT_LOG.md").read_text(encoding="utf-8")
    assert "hf_" not in project_log
    assert "API Key Available" not in project_log

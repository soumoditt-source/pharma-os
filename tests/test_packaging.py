from __future__ import annotations

import math
from pathlib import Path

from server.ml_engine import get_ml_engine


ROOT = Path(__file__).resolve().parents[1]


def test_ml_engine_artifacts_load_and_predict():
    engine = get_ml_engine()
    result = engine.predict_with_uncertainty("CC(=O)Nc1ccc(cc1)O")

    assert math.isfinite(float(result["prediction"]))
    assert math.isfinite(float(result["uncertainty"]))
    low, high = result["confidence_interval"]
    assert low <= high


def test_dockerfile_copies_runtime_artifacts():
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")

    assert "COPY data/ ./data/" in dockerfile
    assert "COPY server/ ./server/" in dockerfile


def test_manifest_dependencies_cover_ml_runtime():
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")

    for dependency in ("numpy", "scikit-learn", "openenv-core"):
        assert dependency in pyproject
        assert dependency in requirements


def test_launch_and_preflight_scripts_exist():
    assert (ROOT / "launch_dashboard.ps1").exists()
    assert (ROOT / "scripts" / "preflight.py").exists()
    assert (ROOT / "scripts" / "validate-submission.sh").exists()

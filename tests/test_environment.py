"""
PharmaOS — Tests
================
Comprehensive test suite for the PharmaOS RL environment.
Covers: property computation, SA score, ADMET proxies, PAINS filter,
task scoring, episode lifecycle, reward function, and edge cases.

Built by: Team Fullstack Shinobi & Soumoditya Das
Event: Meta x PyTorch OpenEnv Hackathon 2026

Run: pytest tests/ -v --tb=short
"""

from __future__ import annotations
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import server.environment as environment_module
from models import PharmaAction, PharmaObservation, PharmaState, AVAILABLE_TASKS
from server.sa_score import compute_sa_score, normalize_sa_score
from server.environment import (
    PharmaEnvironment,
    compute_properties,
    compute_task_score,
    build_admet_summary,
    _estimate_logS,
    _compute_bbb_score,
    _compute_herg_risk,
    _check_pains,
    generate_structure_payload,
    render_mol_svg,
    DRAW_AVAILABLE,
    RDKIT_AVAILABLE,
)

# Well-known molecules for deterministic tests
ASPIRIN   = "CC(=O)Oc1ccccc1C(=O)O"        # MW=180, LogP=1.2, HBD=1, HBA=4
PARACETAMOL = "CC(=O)Nc1ccc(cc1)O"          # MW=151, LogP=0.9, HBD=2, HBA=2
LIPINSKI_VIOLATOR = "CCCCCCCCCCCCCCC(=O)Nc1ccc(cc1)C(=O)Nc1ccc(cc1)N"  # MW > 500
LOW_QED   = "CCCCCCCCCCCC(=O)O"             # Lauric acid, QED ~0.17
BENZENE   = "c1ccccc1"                       # Simplest aromatic
CORONENE  = "c1ccc2ccc3ccc4ccc5ccc6ccccc6c5c4c3c2c1"  # Very large PAH


# ─── SA Score Tests ───────────────────────────────────────────────────────────

class TestSAScore:
    def test_simple_molecule_low_sa(self):
        """Simple molecules should have SA < 3."""
        sa = compute_sa_score(BENZENE)
        assert 1.0 <= sa <= 3.5, f"Benzene SA={sa} should be ≤ 3.5"

    def test_aspirin_moderate_sa(self):
        """Aspirin (known drug) should have moderate SA score."""
        sa = compute_sa_score(ASPIRIN)
        assert 1.0 <= sa <= 5.0, f"Aspirin SA={sa} out of expected range"

    def test_complex_molecule_higher_sa(self):
        """Large/complex molecules should score higher."""
        sa_benzene = compute_sa_score(BENZENE)
        sa_coronene = compute_sa_score(CORONENE)
        assert sa_coronene > sa_benzene, "Coronene should have higher SA than benzene"

    def test_invalid_smiles_returns_10(self):
        """Invalid SMILES should return max score of 10."""
        sa = compute_sa_score("INVALID_SMILES_XYZ")
        assert sa == 10.0

    def test_range_always_valid(self):
        """SA score always in [1, 10]."""
        for smi in [ASPIRIN, PARACETAMOL, BENZENE, LOW_QED, LIPINSKI_VIOLATOR]:
            sa = compute_sa_score(smi)
            assert 1.0 <= sa <= 10.0, f"SA={sa} out of [1,10] for {smi}"

    def test_normalize_sa_score(self):
        """Normalized SA should be in [0, 1] and inverse of raw."""
        assert normalize_sa_score(1.0) == 1.0
        assert normalize_sa_score(10.0) == 0.0
        assert 0.0 < normalize_sa_score(5.0) < 1.0


# ─── Property Computation Tests ───────────────────────────────────────────────

@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="RDKit not installed")
class TestComputeProperties:
    def test_aspirin_properties(self):
        props = compute_properties(ASPIRIN)
        assert props is not None
        assert 170 < props.molecular_weight < 190, f"Aspirin MW={props.molecular_weight}"
        assert props.hbd == 1
        assert props.lipinski_violations == 0

    def test_paracetamol_properties(self):
        props = compute_properties(PARACETAMOL)
        assert props is not None
        assert props.molecular_weight < 200
        assert props.lipinski_violations == 0
        assert props.qed is not None and 0 < props.qed < 1

    def test_lipinski_violator(self):
        props = compute_properties(LIPINSKI_VIOLATOR)
        assert props is not None
        assert props.lipinski_violations >= 1

    def test_invalid_smiles_returns_none(self):
        assert compute_properties("NOT_A_SMILES!!!!") is None

    def test_admet_fields_present(self):
        props = compute_properties(PARACETAMOL)
        assert props is not None
        assert props.logS is not None
        assert props.bbb_score is not None
        assert props.herg_risk is not None
        assert props.pains_alert is not None
        assert props.fsp3 is not None

    def test_fingerprint_similarity_range(self):
        props = compute_properties(ASPIRIN, target_smiles=PARACETAMOL)
        assert props is not None
        assert 0.0 <= props.fingerprint_similarity <= 1.0

    def test_self_similarity_is_one(self):
        props = compute_properties(ASPIRIN, target_smiles=ASPIRIN)
        assert props is not None
        assert props.fingerprint_similarity == pytest.approx(1.0, abs=1e-3)


# ─── ADMET Tests ──────────────────────────────────────────────────────────────

@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="RDKit not installed")
class TestADMET:
    def test_bbb_high_psa_excluded(self):
        """High PSA molecules should have low BBB score."""
        score = _compute_bbb_score(tpsa=150, mw=400, logp=2, hbd=1)
        assert score < 0.6, f"High PSA should give low BBB: {score}"

    def test_bbb_good_profile(self):
        """CNS-drug-like profile should give high BBB score."""
        score = _compute_bbb_score(tpsa=60, mw=300, logp=2.5, hbd=1)
        assert score >= 0.7, f"Good CNS profile should give high BBB: {score}"

    def test_herg_risk_basic_nitrogen(self):
        """Molecules with basic N + high LogP should have higher hERG risk."""
        try:
            from rdkit import Chem
            mol = Chem.MolFromSmiles("CCN")  # Ethylamine
            risk_low = _compute_herg_risk(mol, logp=0.5, mw=100)  # low MW/LogP
            mol2 = Chem.MolFromSmiles("CCCCN")
            risk_high = _compute_herg_risk(mol2, logp=4.5, mw=450)  # high
            assert risk_high >= risk_low
        except ImportError:
            pytest.skip("RDKit not available")

    def test_logS_high_logp_low_solubility(self):
        try:
            from rdkit import Chem
            mol = Chem.MolFromSmiles("CCCCCCCCCCCC")  # dodecane
            logS = _estimate_logS(mol, logp=6.8, mw=170)
            assert logS < -4, f"Lipophilic molecule should have low solubility, got {logS}"
        except ImportError:
            pytest.skip("RDKit not available")

    def test_pains_true_positive(self):
        """Rhodanine is a known PAINS structure."""
        # Rhodanine scaffold
        rhodanine = "O=C1CSC(=S)N1"
        is_pains, desc = _check_pains(
            __import__('rdkit').Chem.MolFromSmiles(rhodanine)
            if RDKIT_AVAILABLE else None
        )
        if RDKIT_AVAILABLE:
            assert is_pains, f"Rhodanine should be flagged as PAINS"

    def test_pains_clean_molecule(self):
        is_pains, _ = _check_pains(
            __import__('rdkit').Chem.MolFromSmiles(PARACETAMOL)
            if RDKIT_AVAILABLE else None
        )
        if RDKIT_AVAILABLE:
            assert not is_pains, "Paracetamol should be PAINS-clean"


# ─── Task Scoring Tests ───────────────────────────────────────────────────────

@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="RDKit not installed")
class TestTaskScoring:
    def test_lipinski_perfect_score(self):
        props = compute_properties(PARACETAMOL)
        assert props is not None
        score = compute_task_score(props, "lipinski_optimizer")
        assert score == pytest.approx(0.99, abs=1e-6), f"Paracetamol should get capped high Lipinski score, got {score}"

    def test_lipinski_violator_low_score(self):
        props = compute_properties(LIPINSKI_VIOLATOR)
        assert props is not None
        score = compute_task_score(props, "lipinski_optimizer")
        assert score < 0.75, f"Lipinski violator should score < 0.75, got {score}"

    def test_qed_score_range(self):
        for smi in [PARACETAMOL, ASPIRIN, BENZENE]:
            props = compute_properties(smi)
            assert props is not None
            score = compute_task_score(props, "qed_optimizer")
            assert 0.0 < score < 1.0, f"QED score must stay in the open interval for {smi}: {score}"

    def test_multi_obj_score_range(self):
        for smi in [PARACETAMOL, ASPIRIN]:
            props = compute_properties(smi, target_smiles=ASPIRIN)
            assert props is not None
            score = compute_task_score(props, "multi_objective_designer")
            assert 0.0 < score < 1.0, f"Multi-obj score must stay in the open interval: {score}"

    def test_pains_penalty_qed(self):
        """PAINS hit should reduce QED task score."""
        from server.environment import _check_pains
        from rdkit import Chem
        rhodanine = "O=C1CSC(=S)N1"
        mol = Chem.MolFromSmiles(rhodanine)
        if mol:
            props = compute_properties(rhodanine)
            if props and props.pains_alert:
                score = compute_task_score(props, "qed_optimizer")
                assert score <= props.qed, "PAINS penalty should reduce QED score"


# ─── Environment Lifecycle Tests ──────────────────────────────────────────────

class TestEnvironmentLifecycle:
    def test_all_tasks_can_reset(self):
        for task in AVAILABLE_TASKS:
            env = PharmaEnvironment(task_name=task)
            obs = env.reset()
            assert isinstance(obs, PharmaObservation)
            assert obs.task_name == task
            assert obs.current_smiles != ""
            assert obs.step_count == 0
            assert not obs.done

    def test_step_after_reset(self):
        env = PharmaEnvironment(task_name="lipinski_optimizer")
        env.reset()
        action = PharmaAction(smiles=PARACETAMOL)
        obs, reward, done, info = env.step(action)
        assert isinstance(obs, PharmaObservation)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert obs.step_count == 1

    def test_invalid_smiles_step(self):
        env = PharmaEnvironment(task_name="lipinski_optimizer")
        env.reset()
        obs, reward, done, info = env.step(PharmaAction(smiles="INVALID_SMILES!!!"))
        assert reward < 0, "Invalid SMILES should give negative reward"
        assert "INVALID" in obs.feedback.upper() or "invalid" in obs.feedback.lower()

    def test_repeat_molecule_penalty(self):
        env = PharmaEnvironment(task_name="qed_optimizer")
        env.reset()
        action = PharmaAction(smiles=ASPIRIN)
        env.step(action)  # First time — novel
        _, reward2, _, info2 = env.step(action)  # Second time — repeat
        assert reward2 < 0, f"Repeat molecule should give negative reward, got {reward2}"
        assert info2.get("is_repeat") or (reward2 <= -0.04)

    def test_episode_ends_at_max_steps(self):
        env = PharmaEnvironment(task_name="lipinski_optimizer")
        env.reset()
        smiles_pool = [ASPIRIN, PARACETAMOL, BENZENE, "Cc1ccccc1", "CC(C)O",
                       "CCCO", "CCC(=O)O", "c1ccncc1", "CC(N)=O", "CCN"]
        done = False
        steps = 0
        while not done and steps < 20:
            smi = smiles_pool[steps % len(smiles_pool)]
            _, _, done, _ = env.step(PharmaAction(smiles=smi))
            steps += 1
        assert done, f"Episode should end by step {env._state.max_steps}"

    def test_get_state_returns_state(self):
        env = PharmaEnvironment(task_name="multi_objective_designer")
        env.reset()
        state = env.get_state()
        assert isinstance(state, PharmaState)
        assert state.task_name == "multi_objective_designer"
        assert len(state.visited_molecules) >= 1

    def test_unknown_task_raises(self):
        with pytest.raises((ValueError, KeyError)):
            PharmaEnvironment(task_name="unknown_task_xyz").reset()

    def test_step_without_reset_raises(self):
        env = PharmaEnvironment(task_name="lipinski_optimizer")
        with pytest.raises(RuntimeError):
            env.step(PharmaAction(smiles=PARACETAMOL))

    def test_history_grows(self):
        env = PharmaEnvironment(task_name="qed_optimizer")
        env.reset()
        smiles = [ASPIRIN, PARACETAMOL, BENZENE]
        for smi in smiles:
            obs, _, _, _ = env.step(PharmaAction(smiles=smi))
        assert len(obs.history) >= 1

    def test_best_score_monotone(self):
        """best_score should never decrease."""
        env = PharmaEnvironment(task_name="qed_optimizer")
        env.reset()
        best = 0.0
        smiles_pool = [PARACETAMOL, ASPIRIN, "Cc1ccccn1", "CC(=O)N", "c1ccoc1"]
        for smi in smiles_pool:
            obs, _, _, _ = env.step(PharmaAction(smiles=smi))
            assert obs.best_score >= best - 1e-6, (
                f"best_score decreased: {best} → {obs.best_score}"
            )
            best = obs.best_score

    def test_step_uses_stub_scoring_when_rdkit_unavailable(self, monkeypatch):
        monkeypatch.setattr(environment_module, "RDKIT_AVAILABLE", False)
        env = environment_module.PharmaEnvironment(task_name="lipinski_optimizer")
        env.reset(seed=101)

        obs, reward, done, info = env.step(PharmaAction(smiles=PARACETAMOL))

        assert isinstance(obs, PharmaObservation)
        assert isinstance(reward, float)
        assert info.get("error") != "invalid_smiles"
        assert obs.current_smiles == PARACETAMOL


# ─── Observation Schema Tests ─────────────────────────────────────────────────

class TestObservationSchema:
    def test_reset_obs_has_all_fields(self):
        env = PharmaEnvironment("lipinski_optimizer")
        obs = env.reset()
        assert obs.task_name == "lipinski_optimizer"
        assert obs.target_description != ""
        assert obs.action_space_description != ""
        assert isinstance(obs.history, list)
        assert isinstance(obs.visited_count, int)
        assert obs.properties is not None

    def test_obs_serializable(self):
        """Observation must be JSON-serializable (for HTTP transport)."""
        import json
        env = PharmaEnvironment("qed_optimizer")
        obs = env.reset()
        data = obs.model_dump()
        json_str = json.dumps(data)
        assert isinstance(json_str, str)
        assert len(json_str) > 100


@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="RDKit not installed")
def test_render_mol_svg_gracefully_handles_missing_draw(monkeypatch):
    from rdkit import Chem

    mol = Chem.MolFromSmiles(PARACETAMOL)
    assert mol is not None

    if DRAW_AVAILABLE:
        assert render_mol_svg(mol) is not None

    monkeypatch.setattr(environment_module, "DRAW_AVAILABLE", False)
    monkeypatch.setattr(environment_module, "rdMolDraw2D", None)
    assert render_mol_svg(mol) is None


@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="RDKit not installed")
def test_generate_structure_payload_returns_2d_block():
    payload = generate_structure_payload(PARACETAMOL)

    assert payload is not None
    assert payload["structure_source"] == "rdkit"
    assert "V2000" in (payload["molblock_2d"] or "")


# ─── Integration: Full Episode ────────────────────────────────────────────────

@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="RDKit not installed")
class TestFullEpisode:
    def test_lipinski_can_achieve_high_score(self):
        """Paracetamol satisfies all Lipinski rules → perfect score."""
        env = PharmaEnvironment("lipinski_optimizer")
        env.reset()
        obs, _, _, _ = env.step(PharmaAction(smiles=PARACETAMOL))
        score = obs.properties.composite_score or compute_task_score(obs.properties, "lipinski_optimizer")
        assert score == pytest.approx(0.99, abs=1e-6), f"Paracetamol should achieve score=0.99, got {score}"

    def test_qed_episode_reward_sum_finite(self):
        env = PharmaEnvironment("qed_optimizer")
        env.reset()
        total_reward = 0.0
        smiles_pool = [PARACETAMOL, ASPIRIN, "Cc1ccccn1", "CC(=O)N", LOW_QED]
        for smi in smiles_pool:
            _, r, done, _ = env.step(PharmaAction(smiles=smi))
            total_reward += r
            if done:
                break
        assert -10 < total_reward < 10, f"Total reward unreasonably large: {total_reward}"

    def test_multi_obj_episode_runs_to_completion(self):
        env = PharmaEnvironment("multi_objective_designer")
        env.reset()
        smiles_pool = [
            PARACETAMOL, ASPIRIN, "c1ccncc1", "CC(=O)N",
            "O=C(N)c1ccccn1", "Cc1ccc(cc1)O", "CC(=O)Nc1ccncc1",
            "Cc1ccncc1", "c1ccoc1", "C1COCCN1", "C1CCNCC1",
            "FC1=CC=CC=C1", "Nc1ccc(cc1)C(=O)N", "CC(C)NC(=O)c1cccnc1",
            "CC(=O)Nc1ccc(F)cc1", "N#Cc1ccncc1", "Cc1ncc(F)cn1",
            "O=C(Nc1ccc(F)cc1)c1ccco1", "CC1=CN=C(N1)c1ccccc1", "c1ccc2ncccc2c1",
        ]
        done = False
        for smi in smiles_pool:
            if done:
                break
            _, _, done, _ = env.step(PharmaAction(smiles=smi))
        assert done, "Multi-obj episode should complete within 20 steps"

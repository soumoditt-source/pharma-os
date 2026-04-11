"""
PharmaOS — Drug Discovery Molecular Optimization RL Environment
server/environment.py: Core RL environment with full RDKit chemistry pipeline.

Scientific pipeline:
  - Lipinski Rule of Five (oral bioavailability)
  - QED (Bickerton et al. 2012) — quantitative drug-likeness
  - SA Score (Ertl & Schuffenhauer 2009) — synthetic accessibility
  - ESOL (Delaney 2004) — aqueous solubility prediction
  - PAINS filter (Baell & Holloway 2010) — pan-assay interference
  - BBB penetration (Di et al. 2003) — CNS drug profile
  - hERG estimation — cardiotoxicity risk
  - Bemis-Murcko scaffold diversity tracking

Built by: Team Fullstack Shinobi & Soumoditya Das
Event: Meta x PyTorch OpenEnv Hackathon 2026
"""

from __future__ import annotations

import math
import uuid
import random
import base64
import io
from typing import Any, Dict, List, Optional, Set, Tuple

# ─── RDKit imports ───────────────────────────────────────────────────────────
try:
    from rdkit import Chem, DataStructs
    from rdkit.Chem import Descriptors, QED, rdMolDescriptors, rdFingerprintGenerator
    from rdkit.Chem.rdMolDescriptors import CalcTPSA, CalcFractionCSP3
    from rdkit.Chem.Scaffolds import MurckoScaffold
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    PAINS_CATALOG = None

try:
    from rdkit.Chem import AllChem
    ALLCHEM_AVAILABLE = True
except ImportError:
    AllChem = None
    ALLCHEM_AVAILABLE = False

try:
    from rdkit.Chem.Draw import rdMolDraw2D
    DRAW_AVAILABLE = True
except ImportError:
    rdMolDraw2D = None
    DRAW_AVAILABLE = False

if RDKIT_AVAILABLE:
    try:
        from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams

        # Build PAINS catalog once at module level.
        _pains_params = FilterCatalogParams()
        _pains_params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
        PAINS_CATALOG = FilterCatalog(_pains_params)
    except ImportError:
        PAINS_CATALOG = None

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import (
    PharmaAction,
    PharmaObservation,
    PharmaState,
    MolecularProperties,
    ADMETSummary,
    TASK_DESCRIPTIONS,
    TASK_SUCCESS_THRESHOLDS,
)
from server.sa_score import compute_sa_score, normalize_sa_score, sa_score_to_label

try:
    from server.ml_engine import get_ml_engine
except Exception:
    def get_ml_engine():
        raise RuntimeError("ML engine unavailable")


# ─── MOLECULE BANKS ──────────────────────────────────────────────────────────
# Carefully selected to cover diverse chemical space and realistic
# drug discovery scenarios

# Task 1: Lipinski violators — each has 2-4 deliberate violations
LIPINSKI_START_MOLECULES = [
    # High MW + High LogP
    "CCCCCCCCCCCCCCC(=O)Nc1ccc(cc1)C(=O)Nc1ccc(cc1)N",
    # Many HBD donors
    "OC(=O)c1ccc(cc1)NC(=O)c1ccc(cc1)NC(=O)c1ccc(cc1)O",
    # High HBA (many ethers)
    "COc1cc(ccc1OC)C(=O)Oc1ccc(OC)c(OC)c1OC",
    # Very high MW peptide-like
    "CC(NC(=O)C(CC(=O)O)NC(=O)C(N)Cc1ccccc1)C(=O)NC(CO)C(=O)O",
    # High LogP fatty chain
    "CCCCCCCCCC(=O)Nc1ccc(cc1)C(=O)NCCCCCCCC",
    # Multiple HBD violations
    "Nc1ccc(cc1)C(=O)Nc1ccc(cc1)C(=O)Nc1ccc(cc1)N",
]

# Task 2: Low QED molecules — starting QED <= 0.25
QED_START_MOLECULES = [
    "CCCCCCCCCCCC(=O)O",              # Lauric acid — QED ~0.17
    "CCCCCCCCC=CCCCCCCCC(=O)O",       # Oleic acid — QED ~0.13
    "c1ccc2c(c1)ccc3cccc4ccc(c4c32)c1cccc2ccccc21",  # Coronene — very low QED
    "CCCCCCCC(=O)OCCO",               # Low QED ester
    "CCCCCCCCC(=O)NCC(=O)O",          # Low QED amino-fatty
    "c1ccc2c(c1)ccc3cccc4ccc(c4c32)c1cccc2ccccc21",  # Coronene — very low QED
]

# Task 3: Simple scaffolds — multi-objective starting points
MULTI_OBJ_START_MOLECULES = [
    "c1ccccc1",        # Benzene
    "C1CCCCC1",        # Cyclohexane
    "c1ccncc1",        # Pyridine
    "c1ccsc1",         # Thiophene
    "c1ccoc1",         # Furan
    "C1CCNCC1",        # Piperidine
]

# Task 3: Reference active molecules (fingerprint similarity targets)
# Selected from public domain drug-like compounds
TARGET_ACTIVE_MOLECULES = [
    "CC1=CC2=C(C=C1)N(C(=O)N2)CC(=O)N1CCN(CC1)C1=CC=C(C=C1)F",  # Drug-like
    "CC(=O)Nc1ccc(cc1)O",        # Paracetamol (acetaminophen)
    "CC(=O)Oc1ccccc1C(=O)O",     # Aspirin
    "CC(Cc1ccc(cc1)CC(C)C(=O)O)C(=O)O",  # Ibuprofen-like
    "Cc1ccc(cc1)S(=O)(=O)N",     # Sulfonamide scaffold
]

TASK_RESET_BASE_SEEDS = {
    "lipinski_optimizer": 101,
    "qed_optimizer": 202,
    "multi_objective_designer": 303,
}

# Medicinal chemistry hint molecules (fallback for inference script)
IMPROVED_MOLECULE_HINTS = {
    "lipinski_optimizer": [
        "CC(=O)Nc1ccc(cc1)O",          # Paracetamol
        "CC(C)(C)NCC(=O)Nc1ccc(C)cc1", # Lidocaine-like
        "Cc1cnn(c1)C(=O)Nc1ccc(F)cc1", # Fluorinated amide
        "c1ccncc1C(=O)O",              # Niacin / Nicotinic acid
        "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O", # Ibuprofen
        "O=C(O)Cc1ccccc1",             # Phenylacetic acid
    ],
    "qed_optimizer": [
        "CC(=O)Nc1ccc(cc1)O",          # Paracetamol
        "Cc1onc(c1)C(=O)Nc1ccccc1",    # Isoxazole amide
        "Cc1cnc(s1)NC(=O)c1ccccc1",    # Thiazole amide
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",# Caffeine (high QED)
        "O=C(Nc1cccnc1)c1ccc(F)cc1",   # Pyridine-fluorophenyl amide
        "C1CCC(CC1)NC(=O)c1ccccc1",    # Cyclohexyl benzamide
    ],
    "multi_objective_designer": [
        "CC(=O)Nc1ccc(cc1)O",          # Paracetamol
        "c1ccc(cc1)CC(=O)N",           # Phenylacetamide
        "CN(C)C(=O)CCS(=O)(=O)c1ccccc1", # Sulfone derivative
        "O=C(Nc1cccc(F)c1)c1ccc(N)cc1",  # Fluoroaniline derivative
        "Cc1ccc(cc1)NC(=O)Cn1ccc2ccccc21", # Indole derivative
        "CC1(C)OC2CC3C4CCC5=CC(=O)CCC5(C)C4CCC3(C)C2(O)O1", # Steroid scaffold
    ],
}


# ─── PROPERTY COMPUTATION ────────────────────────────────────────────────────

def render_mol_svg(mol, size=(300, 200)) -> Optional[str]:
    """Render molecule as SVG string using RDKit."""
    if not RDKIT_AVAILABLE or not DRAW_AVAILABLE or mol is None:
        return None
    try:
        drawer = rdMolDraw2D.MolDraw2DSVG(size[0], size[1])
        drawer.drawOptions().addStereoAnnotation = True
        drawer.drawOptions().addAtomIndices = False
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        svg = drawer.GetDrawingText()
        return svg
    except Exception:
        return None


def generate_structure_payload(smiles: str) -> Optional[Dict[str, Any]]:
    """Build deterministic 2D and 3D structure payloads for the web UI."""
    if not RDKIT_AVAILABLE:
        return None

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    payload: Dict[str, Any] = {
        "smiles": Chem.MolToSmiles(mol),
        "molblock_2d": None,
        "molblock_3d": None,
        "structure_source": "rdkit",
    }

    if ALLCHEM_AVAILABLE:
        try:
            mol2d = Chem.Mol(mol)
            AllChem.Compute2DCoords(mol2d)
            payload["molblock_2d"] = Chem.MolToMolBlock(mol2d)
        except Exception:
            payload["molblock_2d"] = None

        try:
            mol3d = Chem.AddHs(Chem.Mol(mol))
            params = AllChem.ETKDGv3()
            params.randomSeed = 0xF00D
            params.useRandomCoords = False
            status = AllChem.EmbedMolecule(mol3d, params)
            if status == 0:
                try:
                    AllChem.UFFOptimizeMolecule(mol3d, maxIters=200)
                except Exception:
                    pass
                payload["molblock_3d"] = Chem.MolToMolBlock(Chem.RemoveHs(mol3d))
        except Exception:
            payload["molblock_3d"] = None

    return payload


def _estimate_logS(mol, logp: float, mw: float) -> float:
    """
    Estimate aqueous solubility (LogS) using ESOL model.
    Delaney (2004) J. Chem. Inf. Comput. Sci. 44(3):1000-5.
    LogS = 0.16 - 0.63*logP - 0.0062*MW + 0.066*RB - 0.74*AP_fraction
    """
    try:
        rb = rdMolDescriptors.CalcNumRotatableBonds(mol)
        num_aromatic = sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic())
        ap_fraction = num_aromatic / max(mol.GetNumAtoms(), 1)
        logS = 0.16 - 0.63 * logp - 0.0062 * mw + 0.066 * rb - 0.74 * ap_fraction
        return round(logS, 3)
    except Exception:
        return -3.0


def _compute_bbb_score(tpsa: float, mw: float, logp: float, hbd: int) -> float:
    """
    BBB penetration score based on rule-of-thumb criteria.
    Di et al. (2003) -- PAMPA-BBB assay correlation.
    Returns [0, 1] where 1 = ideal CNS-active profile.
    """
    score = 0.0

    # TPSA is the strongest BBB gatekeeper, so it carries the largest weight.
    if tpsa < 90.0:
        score += 0.4
    elif tpsa < 120.0:
        score += 0.2

    if 200.0 <= mw <= 400.0:
        score += 0.2
    elif 150.0 <= mw < 450.0:
        score += 0.1

    if 1.0 <= logp <= 4.0:
        score += 0.2
    elif 0.5 < logp < 5.0:
        score += 0.1

    if hbd <= 2:
        score += 0.1
    elif hbd <= 3:
        score += 0.05

    # Bonus: ideal PSA region
    if 40 <= tpsa <= 80:
        score = min(1.0, score + 0.1)
    return round(score, 4)


def _compute_herg_risk(mol, logp: float, mw: float) -> float:
    """
    Estimate hERG cardiac toxicity risk.
    Based on: Aronov (2008) Drug Discov. Today 13(23-24):1034-41.
    High risk: basic N + MW > 350 + LogP > 3.5
    """
    try:
        # Count basic nitrogens (protonatable at physiological pH)
        basic_n = sum(
            1 for atom in mol.GetAtoms()
            if atom.GetSymbol() == 'N'
            and atom.GetFormalCharge() == 0
            and atom.GetTotalNumHs() > 0
            and atom.GetHybridization().name not in ['SP2']
        )
        if basic_n == 0:
            return 0.1  # Very low risk without basic nitrogen

        risk = 0.3  # Base risk with any basic nitrogen
        if logp > 3.5:
            risk += 0.3
        if mw > 350:
            risk += 0.2
        if basic_n >= 2:
            risk += 0.2

        return round(min(1.0, risk), 4)
    except Exception:
        return 0.3


def _check_pains(mol) -> Tuple[bool, str]:
    """
    Check molecule against PAINS (Pan-Assay Interference) catalog.
    Baell & Holloway (2010) J. Med. Chem. 53(7):2719-40.
    Returns (is_pains, description).
    """
    if not RDKIT_AVAILABLE or PAINS_CATALOG is None or mol is None:
        return False, ""
    try:
        entry = PAINS_CATALOG.GetFirstMatch(mol)
        if entry is not None:
            return True, entry.GetDescription()
        return False, ""
    except Exception:
        return False, ""


def _get_murcko_scaffold(mol) -> str:
    """Get Bemis-Murcko scaffold SMILES for diversity tracking."""
    if not RDKIT_AVAILABLE or mol is None:
        return ""
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaffold) if scaffold else ""
    except Exception:
        return ""


def compute_properties(smiles: str, target_smiles: str = "") -> Optional[MolecularProperties]:
    """
    Compute full molecular property profile using RDKit.
    Returns None if SMILES cannot be parsed.
    """
    if not RDKIT_AVAILABLE:
        return _stub_properties(smiles)

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # ── Basic physicochemical ───────────────────────────────────────────────
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hbd = rdMolDescriptors.CalcNumHBD(mol)
    hba = rdMolDescriptors.CalcNumHBA(mol)
    tpsa = CalcTPSA(mol)
    rot_bonds = rdMolDescriptors.CalcNumRotatableBonds(mol)
    num_heavy = mol.GetNumHeavyAtoms()

    try:
        fsp3 = CalcFractionCSP3(mol)
    except Exception:
        fsp3 = 0.0

    # ── Drug-likeness scores ────────────────────────────────────────────────
    try:
        qed_score = QED.qed(mol)
    except Exception:
        qed_score = 0.0

    sa = compute_sa_score(mol)

    # ── Lipinski violations ─────────────────────────────────────────────────
    violations = sum([mw >= 500, logp >= 5, hbd > 5, hba > 10])

    # ── Fingerprint similarity ──────────────────────────────────────────────
    fp_sim = 0.0
    if target_smiles:
        target_mol = Chem.MolFromSmiles(target_smiles)
        if target_mol is not None:
            morgan_generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
            fp1 = morgan_generator.GetFingerprint(mol)
            fp2 = morgan_generator.GetFingerprint(target_mol)
            fp_sim = DataStructs.TanimotoSimilarity(fp1, fp2)

    # ── ADMET proxies ───────────────────────────────────────────────────────
    logS = _estimate_logS(mol, logp, mw)
    bbb = _compute_bbb_score(tpsa, mw, logp, hbd)
    herg = _compute_herg_risk(mol, logp, mw)
    is_pains, pains_desc = _check_pains(mol)
    
    # ── PyTorch ML Inference ────────────────────────────────────────────────
    try:
        engine = get_ml_engine()
        ml_solubility = engine.predict_solubility(smiles)
        # Normalize prediction approx between 0 to 1 for generic confidence
        ml_confidence = max(0.0, min(1.0, (ml_solubility + 6.0) / 6.0))
    except Exception:
        ml_confidence = 0.5

    # ── Ligand efficiency (proxy) ───────────────────────────────────────────
    le = round(qed_score / max(num_heavy, 1) * 10, 4) if num_heavy > 0 else 0.0

    return MolecularProperties(
        molecular_weight=round(mw, 2),
        logp=round(logp, 3),
        hbd=hbd,
        hba=hba,
        tpsa=round(tpsa, 2),
        rotatable_bonds=rot_bonds,
        num_heavy_atoms=num_heavy,
        qed=round(qed_score, 4),
        sa_score=round(sa, 3),
        fsp3=round(fsp3, 4),
        lipinski_violations=violations,
        fingerprint_similarity=round(fp_sim, 4),
        logS=logS,
        bbb_score=round(bbb, 4),
        herg_risk=round(herg, 4),
        pains_alert=is_pains,
        pains_description=pains_desc if is_pains else None,
        ligand_efficiency=le,
        ml_bioactivity_confidence=round(ml_confidence, 4),
        composite_score=None,  # set by caller
    )


def _stub_properties(smiles: str) -> MolecularProperties:
    """Deterministic stub (no RDKit) for basic testing."""
    h = abs(hash(smiles)) % 10000
    mw = 200.0 + (h % 350)
    logp = round((h % 80) / 20.0 - 1.0, 3)
    hbd = h % 6
    hba = h % 12
    qed = round((h % 900 + 100) / 1000.0, 4)
    sa = round(1.0 + (h % 80) / 10.0, 2)
    violations = sum([mw >= 500, logp >= 5, hbd > 5, hba > 10])
    return MolecularProperties(
        molecular_weight=round(mw, 2), logp=logp, hbd=hbd, hba=hba,
        tpsa=round(50.0 + h % 80, 2), rotatable_bonds=h % 8,
        num_heavy_atoms=h % 30 + 10, qed=qed, sa_score=sa,
        fsp3=round((h % 100) / 100.0, 4), lipinski_violations=violations,
        fingerprint_similarity=round((h % 600) / 1000.0, 4),
        logS=round(-3.0 + (h % 60) / 20.0, 3),
        bbb_score=round((h % 700) / 1000.0, 4),
        herg_risk=round((h % 500) / 1000.0, 4),
        pains_alert=False, ligand_efficiency=round(qed / max(h % 30 + 10, 1) * 10, 4),
    )


def build_admet_summary(props: MolecularProperties) -> ADMETSummary:
    """Build human-readable ADMET summary from properties."""
    logS = props.logS or -3.0
    if logS > -1:
        sol_class = "High"
    elif logS > -3:
        sol_class = "Moderate"
    elif logS > -5:
        sol_class = "Low"
    else:
        sol_class = "Very Low"

    herg = props.herg_risk or 0.0
    herg_class = "Low" if herg < 0.3 else ("Medium" if herg < 0.6 else "High")
    bbb_score = props.bbb_score or 0.0

    # Basic oral bioavailability check
    oral = (
        (props.molecular_weight or 999) < 500
        and (props.logp or 99) < 5
        and (props.hbd or 99) <= 5
        and (props.hba or 99) <= 10
        and (props.tpsa or 999) < 140
    )

    return ADMETSummary(
        logS=logS,
        solubility_class=sol_class,
        bbb_penetrant=bbb_score >= 0.6,
        bbb_score=bbb_score,
        herg_risk=herg,
        herg_class=herg_class,
        pains_alert=props.pains_alert or False,
        oral_bioavailable=oral,
        ml_bioactivity_confidence=props.ml_bioactivity_confidence,
    )


# ─── TASK SCORING ─────────────────────────────────────────────────────────────

STRICT_SCORE_MIN = 0.01
STRICT_SCORE_MAX = 0.99


def _strict_unit_interval(score: float) -> float:
    """Clamp public task scores to the open interval (0, 1)."""
    return round(max(STRICT_SCORE_MIN, min(STRICT_SCORE_MAX, score)), 4)


def compute_task_score(props: MolecularProperties, task_name: str) -> float:
    """Compute composite task score [0.0 – 1.0]."""

    if task_name == "lipinski_optimizer":
        rules_passed = 4 - (props.lipinski_violations or 0)
        base_score = rules_passed / 4.0

        # Penalize severe overshoot so one extreme violation is clearly worse
        # than a near-threshold compound that barely misses a rule.
        severity_penalty = 0.0
        if props.molecular_weight and props.molecular_weight > 500:
            severity_penalty += min(0.12, (props.molecular_weight - 500.0) / 1000.0)
        if props.logp and props.logp > 5:
            severity_penalty += min(0.12, (props.logp - 5.0) * 0.04)
        if props.hbd and props.hbd > 5:
            severity_penalty += min(0.08, (props.hbd - 5) * 0.02)
        if props.hba and props.hba > 10:
            severity_penalty += min(0.08, (props.hba - 10) * 0.01)

        return _strict_unit_interval(base_score - severity_penalty)

    elif task_name == "qed_optimizer":
        qed = props.qed or 0.0
        pains_penalty = 0.20 if (props.pains_alert) else 0.0
        return _strict_unit_interval(qed - pains_penalty)

    elif task_name == "multi_objective_designer":
        qed = props.qed or 0.0
        sa_norm = normalize_sa_score(props.sa_score or 5.0)
        sim = props.fingerprint_similarity or 0.0
        bbb = props.bbb_score or 0.0
        sol_score = _logS_to_score(props.logS or -3.0)
        herg = props.herg_risk or 0.5
        admet = 0.4 * bbb + 0.3 * sol_score + 0.3 * (1.0 - herg)

        # PAINS kills QED component entirely
        if props.pains_alert:
            qed = 0.0

        composite = 0.35 * qed + 0.25 * sa_norm + 0.20 * sim + 0.20 * admet
        return _strict_unit_interval(composite)

    return STRICT_SCORE_MIN


def _logS_to_score(logS: float) -> float:
    """Map LogS to [0,1] score. Good solubility is -2 to 0."""
    if logS > -1:
        return 1.0
    elif logS > -3:
        return 0.8
    elif logS > -5:
        return 0.4
    elif logS > -7:
        return 0.1
    else:
        return 0.0


# ─── ENVIRONMENT CLASS ────────────────────────────────────────────────────────

class PharmaEnvironment:
    """
    OpenEnv-compatible RL environment for drug discovery molecular optimization.

    An AI agent acts as a medicinal chemist, iteratively proposing SMILES
    molecular modifications to improve multi-dimensional drug-likeness scores.

    Three tasks of increasing difficulty:
      1. lipinski_optimizer  (EASY)  — 10 steps
      2. qed_optimizer       (MEDIUM) — 15 steps
      3. multi_objective_designer (HARD) — 20 steps

    Reward function (per step):
      Valid & Novel:  reward = Δscore + 0.02 (novelty) + 0.05 (new scaffold)
      PAINS hit:      reward -= 0.15
      Repeat mol:     reward = -0.05
      Invalid SMILES: reward = -0.10
      Terminal:       bonus = best_episode_score

    Episode ends when:
      - max_steps reached OR
      - task success threshold reached
    """

    MAX_HISTORY = 5  # Steps to include in observation history

    def __init__(self, task_name: str = "lipinski_optimizer"):
        if task_name not in TASK_DESCRIPTIONS:
            raise ValueError(
                f"Unknown task '{task_name}'. "
                f"Must be one of: {list(TASK_DESCRIPTIONS.keys())}"
            )
        self.task_name = task_name
        self._state: Optional[PharmaState] = None
        self._current_smiles: str = ""
        self._target_smiles: str = ""
        self._current_props: Optional[MolecularProperties] = None
        self._current_admet: Optional[ADMETSummary] = None
        self._current_score: float = 0.0
        self._visited: Dict[str, float] = {}
        self._scaffolds: Set[str] = set()
        self._history: List[Dict[str, Any]] = []
        self._reset_counter: int = 0
        self._episode_seed: Optional[int] = None

    # ── Core API ──────────────────────────────────────────────────────────────

    def _select_valid_seed(
        self,
        candidates: List[str],
        target_smiles: str = "",
        rng: Optional[random.Random] = None,
    ) -> Tuple[str, MolecularProperties]:
        """Pick the first starter molecule that can be fully evaluated."""
        ordered: List[str] = list(candidates or [])
        if rng is not None and ordered:
            rng.shuffle(ordered)

        ordered.extend(IMPROVED_MOLECULE_HINTS.get(self.task_name, []))
        ordered.extend([
            "CC(=O)Nc1ccc(cc1)O",
            "CC(=O)Oc1ccccc1C(=O)O",
            "c1ccccc1",
        ])

        seen: Set[str] = set()
        for smiles in ordered:
            if not smiles or smiles in seen:
                continue
            seen.add(smiles)
            props = compute_properties(smiles, target_smiles)
            if props is not None:
                return smiles, props

        raise RuntimeError(
            f"Unable to find a valid starter molecule for task '{self.task_name}'."
        )

    def reset(self, seed: Optional[int] = None) -> PharmaObservation:
        """Start a new episode. Returns the initial observation."""
        episode_id = str(uuid.uuid4())[:8]
        episode_seed = (
            int(seed)
            if seed is not None
            else TASK_RESET_BASE_SEEDS.get(self.task_name, 0) + self._reset_counter
        )
        rng = random.Random(episode_seed)
        self._reset_counter += 1
        self._episode_seed = episode_seed

        if self.task_name == "lipinski_optimizer":
            start_pool = LIPINSKI_START_MOLECULES
            target_smiles = ""
            max_steps = 10
        elif self.task_name == "qed_optimizer":
            start_pool = QED_START_MOLECULES
            target_smiles = ""
            max_steps = 15
        else:
            start_pool = MULTI_OBJ_START_MOLECULES
            target_smiles = rng.choice(TARGET_ACTIVE_MOLECULES)
            max_steps = 20

        start_smiles, props = self._select_valid_seed(start_pool, target_smiles, rng)
        self._current_smiles = start_smiles
        self._target_smiles = target_smiles
        self._visited = {}
        self._scaffolds = set()
        self._history = []

        initial_score = compute_task_score(props, self.task_name)
        props.composite_score = initial_score
        admet = build_admet_summary(props)

        self._current_props = props
        self._current_admet = admet
        self._current_score = initial_score
        self._visited[start_smiles] = initial_score

        # Track scaffold
        mol = Chem.MolFromSmiles(start_smiles) if RDKIT_AVAILABLE else None
        scaffold = _get_murcko_scaffold(mol) if mol else ""
        if scaffold:
            self._scaffolds.add(scaffold)

        self._state = PharmaState(
            episode_id=episode_id,
            step_count=0,
            task_name=self.task_name,
            max_steps=max_steps,
            initial_smiles=start_smiles,
            target_smiles=target_smiles,
            best_score=initial_score,
            best_smiles=start_smiles,
            visited_molecules=[start_smiles],
            done=False,
            reward_history=[],
            unique_scaffolds=len(self._scaffolds),
            metadata={"episode_seed": episode_seed},
        )

        hist_entry = {"smiles": start_smiles, "score": initial_score, "step": 0}
        self._history.append(hist_entry)

        svg = render_mol_svg(mol) if mol else None
        feedback = self._build_reset_feedback(props, admet, initial_score, start_smiles)

        return PharmaObservation(
            done=False,
            reward=None,
            current_smiles=start_smiles,
            properties=props,
            admet=admet,
            feedback=feedback,
            step_count=0,
            task_name=self.task_name,
            target_description=TASK_DESCRIPTIONS[self.task_name],
            action_space_description=(
                "Send a JSON action: {\"smiles\": \"<SMILES>\", \"reasoning\": \"<optional>\"}. "
                "The SMILES must be a valid, chemically sensible modification of the current molecule."
            ),
            visited_count=1,
            best_score=initial_score,
            history=list(self._history[-self.MAX_HISTORY:]),
            mol_svg=svg,
            metadata={
                "episode_seed": episode_seed,
                **({"target_smiles": target_smiles} if target_smiles else {}),
            },
        )

    def step(
        self, action: PharmaAction
    ) -> Tuple[PharmaObservation, float, bool, Dict[str, Any]]:
        """Execute one step. Returns (observation, reward, done, info)."""
        if self._state is None:
            raise RuntimeError("Call reset() before step().")

        self._state.step_count += 1
        step = self._state.step_count
        max_steps = self._state.max_steps
        prev_score = self._current_score
        prev_smiles = self._current_smiles

        proposed = action.smiles.strip()

        # ── Validate SMILES ───────────────────────────────────────────────
        mol = Chem.MolFromSmiles(proposed) if RDKIT_AVAILABLE else None
        # Always go through compute_properties so the deterministic stub fallback
        # still works when RDKit is unavailable in lightweight container builds.
        props = compute_properties(proposed, self._target_smiles)

        if props is None:
            reward = -0.10
            self._state.invalid_attempts += 1
            self._state.reward_history.append(reward)
            done = step >= max_steps
            if done:
                self._state.done = True
            feedback = (
                f"❌ Step {step}: INVALID SMILES — '{proposed[:50]}'\n"
                f"   RDKit could not parse this structure. Penalty: {reward:.2f}\n"
                f"   Current molecule unchanged: {prev_smiles}\n"
                f"   💡 Tip: Make incremental changes. Start from: {prev_smiles}"
            )
            return self._make_obs(
                prev_smiles, self._current_props or MolecularProperties(),
                self._current_admet, reward, done, step, feedback
            ), reward, done, {"error": "invalid_smiles", "step": step}

        # ── Valid molecule ────────────────────────────────────────────────
        admet = build_admet_summary(props)
        new_score = compute_task_score(props, self.task_name)
        props.composite_score = new_score
        is_repeat = proposed in self._visited

        if is_repeat:
            reward = -0.05
            feedback = (
                f"⚠️ Step {step}: REPEAT molecule — already tried this one.\n"
                f"   Previous score: {self._visited[proposed]:.4f} | Penalty: {reward:.2f}\n"
                f"   💡 Tip: Explore novel chemical space — try a different functional group."
            )
            done = step >= max_steps
        else:
            # ── Novel molecule reward ─────────────────────────────────────
            improvement = new_score - prev_score
            reward = improvement + 0.02  # novelty bonus

            # Scaffold diversity bonus
            scaffold = _get_murcko_scaffold(mol) if mol else ""
            scaffold_bonus = 0.0
            if scaffold and scaffold not in self._scaffolds:
                self._scaffolds.add(scaffold)
                scaffold_bonus = 0.05
                reward += scaffold_bonus
                self._state.unique_scaffolds = len(self._scaffolds)

            # PAINS penalty
            pains_penalty = 0.0
            if props.pains_alert:
                pains_penalty = -0.15
                reward += pains_penalty
                self._state.pains_attempts += 1

            reward = round(max(-0.5, min(1.0, reward)), 4)

            # Update tracked state
            if new_score > self._state.best_score:
                self._state.best_score = new_score
                self._state.best_smiles = proposed

            self._current_smiles = proposed
            self._current_props = props
            self._current_admet = admet
            self._current_score = new_score
            self._visited[proposed] = new_score
            self._state.visited_molecules.append(proposed)
            self._history.append({"smiles": proposed, "score": new_score, "step": step})

            feedback = self._build_step_feedback(
                step, proposed, props, admet, new_score, prev_score,
                improvement, reward, scaffold_bonus, pains_penalty
            )

            done = step >= max_steps or self._is_solved(new_score)

        self._state.reward_history.append(reward)

        if done:
            self._state.done = True
            feedback += (
                f"\n\n🏁 EPISODE COMPLETE after {step} steps\n"
                f"   Best Score:    {self._state.best_score:.4f}\n"
                f"   Best Molecule: {self._state.best_smiles}\n"
                f"   Unique Mols:   {len(self._visited)}\n"
                f"   Unique Scaffolds: {len(self._scaffolds)}"
            )

        svg = render_mol_svg(mol) if (mol and not is_repeat) else None

        return self._make_obs(
            self._current_smiles,
            self._current_props or MolecularProperties(),
            self._current_admet,
            reward, done, step, feedback, svg
        ), reward, done, {
            "new_score": new_score,
            "prev_score": prev_score,
            "best_score": self._state.best_score,
            "step": step,
            "is_repeat": is_repeat,
            "is_valid": True,
            "scaffolds_explored": len(self._scaffolds),
        }

    def get_state(self) -> PharmaState:
        if self._state is None:
            return PharmaState()
        return self._state

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _is_solved(self, score: float) -> bool:
        return score >= TASK_SUCCESS_THRESHOLDS.get(self.task_name, STRICT_SCORE_MAX)

    def _make_obs(
        self, smiles, props, admet, reward, done, step, feedback, svg=None
    ) -> PharmaObservation:
        # Determine reasoning hint via agent (only quick RAG lookup to avoid delay)
        # Safe fallback logic
        reasoning_hint = None
        if not done and self._state and step >= 0:
            try:
                from server.agent import agent as pharma_agent
                query = "How can I improve my drug score here? Give a short rule of thumb."
                # Quick cache/rag logic - we access internal methods to be extremely fast and avoid LLM delay.
                trace = pharma_agent._fast_cache_match(query)
                if not trace: 
                    trace = pharma_agent._rag_search(query)
                if trace:
                    reasoning_hint = trace.recommendation
            except Exception:
                pass
                
        return PharmaObservation(
            done=done,
            reward=reward,
            current_smiles=smiles,
            properties=props,
            admet=admet,
            feedback=feedback,
            reasoning_hint=reasoning_hint,
            step_count=step,
            task_name=self.task_name,
            target_description=TASK_DESCRIPTIONS[self.task_name],
            action_space_description='{"smiles": "<SMILES>", "reasoning": "<optional>"}',
            visited_count=len(self._visited),
            best_score=self._state.best_score if self._state else 0.0,
            history=list(self._history[-self.MAX_HISTORY:]),
            mol_svg=svg,
            metadata={"episode_seed": self._episode_seed} if self._episode_seed is not None else {},
        )

    def _build_reset_feedback(
        self, props: MolecularProperties, admet: ADMETSummary,
        score: float, smiles: str
    ) -> str:
        def fmt_float(value: Optional[float], digits: int = 4, suffix: str = "") -> str:
            if value is None:
                return f"-{suffix}"
            return f"{value:.{digits}f}{suffix}"

        def fmt_int(value: Optional[int], suffix: str = "") -> str:
            if value is None:
                return f"-{suffix}"
            return f"{value}{suffix}"

        sa_label = sa_score_to_label(props.sa_score or 5.0)
        lines = [
            f"🔬 PharmaOS — {self.task_name} — Episode Started",
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            f"📌 Starting Molecule: {smiles}",
            "",
            "📊 Molecular Properties:",
            f"   MW:    {fmt_float(props.molecular_weight, 2, ' Da')}    | LogP: {fmt_float(props.logp, 3)}",
            f"   HBD:   {fmt_int(props.hbd)}              | HBA:  {fmt_int(props.hba)}",
            f"   QED:   {props.qed:.4f}         | TPSA: {props.tpsa} Å²",
            f"   SA:    {fmt_float(props.sa_score, 2)} ({sa_label}) | Fsp3: {fmt_float(props.fsp3, 4)}",
            f"   RotBonds: {fmt_int(props.rotatable_bonds)}       | HeavyAtoms: {fmt_int(props.num_heavy_atoms)}",
            f"   Lipinski Violations: {fmt_int(props.lipinski_violations)}/4",
        ]
        if props.fingerprint_similarity and props.fingerprint_similarity > 0:
            lines.append(f"   Tanimoto Similarity to Target: {props.fingerprint_similarity:.4f}")
        lines += [
            "",
            "🧪 ADMET Profile:",
            f"   Solubility (LogS): {props.logS:.2f} → {admet.solubility_class}",
            f"   BBB Score: {props.bbb_score:.4f} | {'✅ CNS-penetrant' if admet.bbb_penetrant else '❌ CNS-excluded'}",
            f"   hERG Risk: {props.herg_risk:.4f} → {admet.herg_class}",
            f"   PAINS: {'⚠️ ALERT — ' + (props.pains_description or '') if props.pains_alert else '✅ CLEAN'}",
            f"   Oral Bioavailable: {'✅ YES' if admet.oral_bioavailable else '❌ NO'}",
            "",
            f"🎯 Initial Score: {score:.4f} / 1.0",
            "",
            TASK_DESCRIPTIONS[self.task_name],
            "",
            "💡 Propose a SMILES modification to improve your score.",
        ]
        return "\n".join(lines)

    def _build_step_feedback(
        self, step, smiles, props, admet, new_score, prev_score,
        improvement, reward, scaffold_bonus, pains_penalty
    ) -> str:
        delta_sym = "📈" if improvement >= 0 else "📉"
        sa_label = sa_score_to_label(props.sa_score or 5.0)
        lines = [
            f"Step {step}: {delta_sym} Score {prev_score:.4f} → {new_score:.4f} "
            f"(Δ{improvement:+.4f}) | Reward: {reward:+.4f}",
            f"📌 New Molecule: {smiles}",
            "",
            "📊 Properties:",
            f"   MW: {props.molecular_weight} Da | LogP: {props.logp} | HBD: {props.hbd} | HBA: {props.hba}",
            f"   QED: {props.qed:.4f} | TPSA: {props.tpsa} Å² | SA: {props.sa_score:.2f} ({sa_label})",
            f"   Fsp3: {props.fsp3:.4f} | Lipinski Violations: {props.lipinski_violations}/4",
        ]
        if props.fingerprint_similarity and self._target_smiles:
            lines.append(f"   Tanimoto Similarity: {props.fingerprint_similarity:.4f}")
        lines += [
            "",
            f"🧪 ADMET: LogS={props.logS:.2f} | BBB={props.bbb_score:.3f} | hERG={props.herg_risk:.3f}",
        ]
        if pains_penalty < 0:
            lines.append(f"   ⚠️ PAINS ALERT: {props.pains_description} | Penalty: {pains_penalty:.2f}")
        if scaffold_bonus > 0:
            lines.append(f"   🔎 New scaffold explored! Diversity bonus: +{scaffold_bonus:.2f}")
        lines.append("")
        if new_score > prev_score:
            lines.append("✅ Improvement! Keep pushing.")
        elif abs(new_score - prev_score) < 1e-6:
            lines.append("➡️  No score change. Try a different modification.")
        else:
            lines.append("⚠️  Score decreased. Try reverting or a different direction.")
        lines.append(f"🏆 Best Score This Episode: {self._state.best_score:.4f}")
        return "\n".join(lines)

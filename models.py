"""
PharmaOS — Drug Discovery Molecular Optimization RL Environment
models.py: Type-safe Pydantic models for Action, Observation, State, ADMET.

Scientific grounding:
  - Lipinski (2001) Rule of Five — oral bioavailability
  - Bickerton et al. (2012) QED — quantitative drug-likeness
  - Ertl & Schuffenhauer (2009) SA Score — synthetic accessibility
  - Delaney (2004) ESOL — aqueous solubility
  - Di et al. (2003) PAMPA-BBB — BBB penetration
  - Ertl et al. (2006) PAINS — pan-assay interference

Built by: Team Fullstack Shinobi & Soumoditya Das
Event: Meta x PyTorch OpenEnv Hackathon 2026
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# ACTION MODEL
# ---------------------------------------------------------------------------

class PharmaAction(BaseModel):
    """
    An action the agent takes: propose a new molecular SMILES string.

    The agent acts as a medicinal chemist, proposing targeted structural
    modifications to improve the molecule's drug-likeness profile.

    Attributes:
        smiles: SMILES string of the proposed molecule
        reasoning: Optional chain-of-thought reasoning (not graded, aids debug)
        metadata: Arbitrary extra fields
    """
    smiles: str = Field(
        ...,
        description="SMILES string of the proposed molecule",
        examples=["CC(=O)Nc1ccc(cc1)O", "CC(=O)Oc1ccccc1C(=O)O"],
    )
    reasoning: str = Field(
        default="",
        description="Optional reasoning/explanation from the agent",
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# MOLECULAR PROPERTIES MODEL
# ---------------------------------------------------------------------------

class MolecularProperties(BaseModel):
    """
    Computed molecular properties via RDKit.

    Covers the key ADME/drug-likeness metrics used in hit-to-lead
    and lead optimization phases of drug discovery.
    """
    # Primary physicochemical properties
    molecular_weight: Optional[float] = Field(None, description="Molecular weight in Da (target: < 500)")
    logp: Optional[float] = Field(None, description="Wildman-Crippen LogP (target: < 5, ideally 1-3)")
    hbd: Optional[int] = Field(None, description="H-bond donors (target: ≤ 5)")
    hba: Optional[int] = Field(None, description="H-bond acceptors (target: ≤ 10)")
    tpsa: Optional[float] = Field(None, description="Topological Polar Surface Area in Å² (oral: < 140)")
    rotatable_bonds: Optional[int] = Field(None, description="Rotatable bonds (oral: < 10)")
    num_heavy_atoms: Optional[int] = Field(None, description="Number of heavy atoms")

    # Drug-likeness scores
    qed: Optional[float] = Field(None, description="Quantitative Estimate of Drug-likeness [0–1]")
    sa_score: Optional[float] = Field(None, description="Synthetic Accessibility Score [1=easy, 10=hard]")
    fsp3: Optional[float] = Field(None, description="Fraction of sp3 carbons [0–1] — 3D drug-likeness")

    # Lipinski
    lipinski_violations: Optional[int] = Field(None, description="Lipinski Rule of Five violations [0–4]")

    # Fingerprint similarity to reference
    fingerprint_similarity: Optional[float] = Field(
        None, description="Tanimoto similarity to target reference molecule [0–1]"
    )

    # ADMET proxies
    logS: Optional[float] = Field(None, description="Estimated aqueous solubility LogS (ESOL model)")
    bbb_score: Optional[float] = Field(
        None, description="BBB penetration score [0–1] based on pKa/PSA/MW/LogP rules"
    )
    herg_risk: Optional[float] = Field(
        None, description="Estimated hERG toxicity risk [0–1]; lower is safer"
    )
    pains_alert: Optional[bool] = Field(
        None, description="PAINS filter hit — pan-assay interference compound flag"
    )
    pains_description: Optional[str] = Field(
        None, description="Description of PAINS pattern matched (if any)"
    )

    # High-End ML Predictions
    ml_bioactivity_confidence: Optional[float] = Field(
        None, description="PyTorch Deep Neural Network prediction for biological activity / solubility [0-100%]"
    )

    # Composite
    ligand_efficiency: Optional[float] = Field(
        None, description="Ligand Efficiency = QED / num_heavy_atoms * 10 (proxy)"
    )
    composite_score: Optional[float] = Field(
        None, description="Task-specific composite score [0–1]"
    )


# ---------------------------------------------------------------------------
# ADMET SUMMARY MODEL
# ---------------------------------------------------------------------------

class ADMETSummary(BaseModel):
    """
    Concise ADMET (Absorption, Distribution, Metabolism, Excretion, Toxicity)
    property summary. All values are in-silico estimates.
    """
    logS: Optional[float] = Field(None, description="LogS aqueous solubility estimate")
    solubility_class: Optional[str] = Field(None, description="Solubility class: High/Moderate/Low/Very Low")
    bbb_penetrant: Optional[bool] = Field(None, description="Predicted CNS-active / BBB penetrant")
    bbb_score: Optional[float] = Field(None, description="BBB score [0-1], higher = better CNS profile")
    herg_risk: Optional[float] = Field(None, description="hERG inhibition risk estimate [0-1]")
    herg_class: Optional[str] = Field(None, description="hERG risk: Low/Medium/High")
    pains_alert: bool = Field(default=False, description="Pan-Assay Interference flag")
    oral_bioavailable: Optional[bool] = Field(None, description="Passes basic oral bioavailability criteria")
    ml_bioactivity_confidence: Optional[float] = Field(None, description="Confidence score from Deep Learning Predictor [0-1]")


# ---------------------------------------------------------------------------
# OBSERVATION MODEL
# ---------------------------------------------------------------------------

class PharmaObservation(BaseModel):
    """
    What the agent observes after each step or reset.

    The observation contains the full molecular property profile,
    structured feedback, episode state, and (optionally) a rendered
    SVG of the molecule for visual inspection.
    """
    done: bool = Field(default=False)
    reward: Optional[float] = Field(default=None, description="Reward for last action")
    current_smiles: str = Field(default="", description="Active molecule SMILES")
    properties: MolecularProperties = Field(default_factory=MolecularProperties)
    admet: Optional[ADMETSummary] = Field(default=None, description="ADMET property summary")
    feedback: str = Field(default="", description="Human-readable step feedback")
    reasoning_hint: Optional[str] = Field(default=None, description="Optional reasoning chain or hints generated by the agent")
    step_count: int = Field(default=0)
    task_name: str = Field(default="")
    target_description: str = Field(default="")
    action_space_description: str = Field(
        default="Provide a SMILES string of your proposed molecule modification.",
    )
    visited_count: int = Field(default=0, description="Unique molecules tried so far")
    best_score: float = Field(default=0.0, description="Best composite score this episode")
    history: List[Dict[str, Any]] = Field(default_factory=list, description="Recent mol/score history (last 5)")
    mol_svg: Optional[str] = Field(
        default=None, description="SVG rendering of the current molecule (base64 or raw SVG)"
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# STATE (EPISODE METADATA) MODEL
# ---------------------------------------------------------------------------

class PharmaState(BaseModel):
    """
    Episode-level metadata. Provides complete bookkeeping state.
    """
    episode_id: Optional[str] = Field(default=None)
    step_count: int = Field(default=0)
    task_name: str = Field(default="")
    max_steps: int = Field(default=10)
    initial_smiles: str = Field(default="")
    target_smiles: str = Field(default="")
    best_score: float = Field(default=0.0)
    best_smiles: str = Field(default="")
    visited_molecules: List[str] = Field(default_factory=list)
    done: bool = Field(default=False)
    reward_history: List[float] = Field(default_factory=list)
    unique_scaffolds: int = Field(default=0, description="Number of unique Bemis-Murcko scaffolds explored")
    pains_attempts: int = Field(default=0, description="Number of PAINS molecules proposed")
    invalid_attempts: int = Field(default=0, description="Number of invalid SMILES proposed")
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# TASK REGISTRY
# ---------------------------------------------------------------------------

AVAILABLE_TASKS = [
    "lipinski_optimizer",         # EASY: satisfy Lipinski Rule of Five
    "qed_optimizer",              # MEDIUM: maximize QED drug-likeness
    "multi_objective_designer",   # HARD: multi-property + ADMET optimization
]

TASK_DESCRIPTIONS = {
    "lipinski_optimizer": (
        "🟢 EASY — Lipinski Rule of Five Optimizer\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "OBJECTIVE: Modify the molecule to satisfy ALL 4 Lipinski rules\n"
        "for oral drug bioavailability.\n\n"
        "Rules:\n"
        "  1. Molecular Weight (MW) < 500 Da\n"
        "  2. LogP (partition coefficient) < 5\n"
        "  3. H-Bond Donors (HBD) ≤ 5\n"
        "  4. H-Bond Acceptors (HBA) ≤ 10\n\n"
        "Score: rules_satisfied / 4, clamped to the open interval (0, 1)\n"
        "Target: Score >= 0.99 (all 4 rules satisfied without endpoint values)\n"
        "Max steps: 10\n\n"
        "Strategy: Remove carbon chains to reduce MW, add polar groups to\n"
        "reduce LogP, replace NH/OH groups to reduce HBD/HBA count."
    ),
    "qed_optimizer": (
        "🟡 MEDIUM — QED Drug-Likeness Optimizer\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "OBJECTIVE: Maximize QED (Quantitative Estimate of Drug-likeness).\n"
        "QED integrates MW, LogP, HBD, HBA, PSA, rotatable bonds, and\n"
        "aromatic rings into a desirability function.\n\n"
        "Score: current QED, clamped to the open interval (0, 1)\n"
        "Starting QED: ≤ 0.3 | Target: QED ≥ 0.75\n"
        "Max steps: 15\n\n"
        "PAINS Filter: Molecules matching Pan-Assay Interference patterns\n"
        "receive a -0.20 penalty per step.\n\n"
        "Strategy: Balance ALL properties simultaneously. Common drug-like\n"
        "fragments: phenyl, pyridine, morpholine, piperazine, amide bonds.\n"
        "Ideal QED region: MW 300-400, LogP 2-4, 1-2 aromatic rings."
    ),
    "multi_objective_designer": (
        "🔴 HARD — Multi-Parameter Optimization (MPO)\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "OBJECTIVE: Simultaneously optimize multiple drug design criteria\n"
        "including drug-likeness, synthesizability, bioactivity, and ADMET.\n\n"
        "Composite Score = 0.35×QED + 0.25×SA_norm + 0.20×Similarity + 0.20×ADMET\n"
        "  where:\n"
        "  • SA_norm = (10 - SA_score) / 9   (higher = easier synthesis)\n"
        "  • Similarity = Tanimoto to known bioactive molecule\n"
        "  • ADMET = 0.4×BBB_score + 0.3×Solubility_score + 0.3×(1 - hERG_risk)\n\n"
        "PAINS Alert: Automatically zeros out the QED component\n"
        "Score: Composite, clamped to the open interval (0, 1)\n"
        "Target: Score ≥ 0.70\n"
        "Max steps: 20\n\n"
        "Strategy: Use drug-like scaffolds (indole, benzimidazole, pyrimidine).\n"
        "Avoid long alkyl chains (↑ LogP, ↓ solubility, ↑ hERG risk).\n"
        "Aim for PSA < 90 Å² for CNS activity, or 60-120 Å² for oral drugs."
    ),
}

TASK_SUCCESS_THRESHOLDS = {
    "lipinski_optimizer": 0.99,   # All rules satisfied without hitting score endpoint
    "qed_optimizer": 0.75,        # Excellent drug-likeness
    "multi_objective_designer": 0.70,  # Strong multi-objective
}

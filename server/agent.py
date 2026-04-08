from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from enum import Enum
from typing import List, Optional

from openai import OpenAI
from server.compound_lookup import resolve_compound_query

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False


class ThinkingLevel(Enum):
    FAST = "FAST_CACHE"
    RAG = "RAG_RETRIEVAL"
    LLM = "LLM_ORACLE"


@dataclass
class ChainOfThought:
    level: str
    confidence: float
    reasoning: str
    recommendation: str
    formula: Optional[str] = None
    provider: Optional[str] = None

    def to_dict(self):
        return asdict(self)


PHARMA_RAG_CORPUS = [
    {
        "keywords": ["solubility", "logs", "esol", "hydrophilic", "water"],
        "formula": "Lower LogP and MW generally improve LogS",
        "knowledge": (
            "Aqueous solubility often improves when lipophilicity and molecular size fall. "
            "Adding polar groups such as alcohols, amines, or acids can raise solubility, "
            "and replacing flat aromatic bulk with less lipophilic motifs can help further."
        ),
    },
    {
        "keywords": ["qed", "drug-likeness", "drug", "optimal"],
        "formula": "QED rewards balanced oral-drug-like property ranges",
        "knowledge": (
            "QED usually improves when MW, LogP, HBD, HBA, PSA, and aromaticity stay in "
            "balanced medicinal-chemistry ranges. Extremely large, greasy, or reactive "
            "structures tend to score poorly."
        ),
    },
    {
        "keywords": ["lipophilicity", "logp", "hydrophobic", "greasy", "water-insoluble"],
        "formula": "LogP rises with hydrophobic fragments and falls with heteroatoms",
        "knowledge": (
            "To lower LogP, add heteroatoms or trim hydrophobic chains and fused aromatics. "
            "To raise LogP, add lipophilic substituents such as halogens or alkyl groups. "
            "A moderate oral range is often around LogP 1 to 3."
        ),
    },
    {
        "keywords": ["hba", "hbd", "hydrogen bond", "acceptor", "donor"],
        "formula": "Lipinski: HBD <= 5 and HBA <= 10",
        "knowledge": (
            "Too many donors or acceptors often hurt permeability. Donor count can be reduced "
            "by masking or removing NH and OH groups, while acceptor count can fall by "
            "simplifying heavily heteroatom-rich motifs."
        ),
    },
    {
        "keywords": ["pains", "interference", "assay", "false positive"],
        "formula": "PAINS hits should be treated as structural liabilities",
        "knowledge": (
            "PAINS motifs can create misleading assay activity. Quinones, rhodanines, and "
            "similar reactive motifs are common offenders and should usually be replaced "
            "before trusting an apparent score improvement."
        ),
    },
    {
        "keywords": ["admet", "bbb", "cns", "brain", "permeability"],
        "formula": "BBB tends to favor lower PSA, moderate LogP, and moderate MW",
        "knowledge": (
            "Blood-brain barrier penetration is usually easier when PSA is low, MW is not too "
            "large, and lipophilicity is moderate. If CNS exposure is undesirable, increasing "
            "polarity is a common way to push compounds out of that window."
        ),
    },
    {
        "keywords": ["herg", "cardiotoxicity", "qt prolongation", "heart"],
        "formula": "hERG risk often rises with basicity and lipophilicity",
        "knowledge": (
            "hERG liability often grows when compounds are both lipophilic and strongly basic. "
            "Reducing amine basicity or lowering LogP can help lower risk."
        ),
    },
    {
        "keywords": ["sa", "synthetic accessibility", "synthesis", "chiral", "complexity"],
        "formula": "SA score worsens with structural complexity",
        "knowledge": (
            "Synthetic accessibility generally improves when stereochemical burden, bridged ring "
            "systems, and unusual fragments are reduced. Using common medicinal-chemistry "
            "building blocks usually helps."
        ),
    },
    {
        "keywords": ["tpsa", "psa", "polar surface area"],
        "formula": "TPSA is a permeability gatekeeper",
        "knowledge": (
            "TPSA is a strong proxy for permeability. Lower TPSA usually helps passive "
            "membrane crossing, while higher TPSA often improves solubility at the cost of "
            "permeability."
        ),
    },
    {
        "keywords": ["mw", "molecular weight", "size"],
        "formula": "High MW often hurts developability",
        "knowledge": (
            "Molecular weight is one of the easiest ways to lose oral drug-likeness. Removing "
            "non-essential fragments or switching to a leaner scaffold can help quickly."
        ),
    },
    {
        "keywords": ["detergent", "soap", "surfactant", "cleaner", "household"],
        "formula": "Household products are often mixtures, not single molecules",
        "knowledge": (
            "Terms like detergent, soap, perfume, and sunscreen usually refer to mixtures or "
            "product classes rather than one exact compound. The safest scientific answer is to "
            "name representative active ingredients or surfactants and explain that the product "
            "itself is not a single molecule."
        ),
    },
    {
        "keywords": ["common name", "brand name", "generic", "everyday chemical"],
        "formula": "A common name may map to a family, alias, or formulation",
        "knowledge": (
            "Common-language chemical names are often ambiguous. A good resolver checks for "
            "generic names, brand-generic synonyms, and representative compounds before assuming "
            "there is one exact SMILES string."
        ),
    },
]


class PharmaAgent:
    """
    Lightweight multi-tier reasoning helper for the dashboard.

    LLM calls use the OpenAI client configured by API_BASE_URL, MODEL_NAME,
    and HF_TOKEN, while local fast-path and retrieval reasoning remain
    available even without a token.
    """

    def __init__(self):
        self.provider_mode = os.getenv("LLM_PROVIDER", "auto").strip().lower()
        self.hf_token = os.getenv("HF_TOKEN")
        self.api_base_url = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
        self.model_name = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
        self.mistral_api_key = os.getenv("MISTRAL_API_KEY")
        self.mistral_api_base_url = os.getenv("MISTRAL_API_BASE_URL", "https://api.mistral.ai/v1")
        self.mistral_model_name = os.getenv("MISTRAL_MODEL_NAME", "mistral-small-latest")
        self.llm_backends = self._build_llm_backends()
        self._init_rag()

    def _build_llm_backends(self) -> List[dict]:
        if self.provider_mode in {"off", "none", "disabled"}:
            return []

        backends: List[dict] = []

        if self.provider_mode in {"auto", "huggingface", "hf"} and self.hf_token:
            backends.append(
                {
                    "provider": "huggingface-router",
                    "client": OpenAI(base_url=self.api_base_url, api_key=self.hf_token),
                    "model": self.model_name,
                }
            )

        if self.provider_mode in {"auto", "mistral"} and self.mistral_api_key:
            backends.append(
                {
                    "provider": "mistral",
                    "client": OpenAI(base_url=self.mistral_api_base_url, api_key=self.mistral_api_key),
                    "model": self.mistral_model_name,
                }
            )

        return backends

    def _init_rag(self):
        self.corpus = PHARMA_RAG_CORPUS
        self.vectorizer = None
        self.tfidf_matrix = None
        if SKLEARN_AVAILABLE:
            self.vectorizer = TfidfVectorizer(stop_words="english")
            docs = [" ".join(entry["keywords"] + entry["knowledge"].split()) for entry in self.corpus]
            self.tfidf_matrix = self.vectorizer.fit_transform(docs)

    def _fast_cache_match(self, query: str) -> Optional[ChainOfThought]:
        q = query.lower()
        if "smiles" in q or "how " in q:
            return ChainOfThought(
                level=ThinkingLevel.FAST.value,
                confidence=1.0,
                reasoning="Direct SMILES guidance requested.",
                recommendation=(
                    "SMILES uses atoms and bonds as text. Use uppercase atoms for aliphatic "
                    "atoms, lowercase c for aromatic carbons, parentheses for branches, and "
                    "symbols like = for multiple bonds. For example, c1ccccc1 is benzene."
                ),
                formula="SMILES Grammar",
            )
        return None

    def _compound_lookup_match(self, query: str) -> Optional[ChainOfThought]:
        lowered = query.lower().strip()
        chemistry_terms = {
            "qed",
            "lipinski",
            "herg",
            "tpsa",
            "admet",
            "solubility",
            "logp",
            "smiles",
            "score",
            "optimize",
            "optimizer",
        }
        if any(term in lowered for term in chemistry_terms):
            return None
        if len(lowered.split()) > 8:
            return None

        payload = resolve_compound_query(query)
        results = payload.get("results", [])
        if not results:
            return None

        top_results = results[:3]
        names = ", ".join(result["name"] for result in top_results)
        loadable = sum(1 for result in top_results if result.get("loadable"))
        explanation = payload.get("summary", "Validated compounds were found for this query.")
        recommendation = (
            f"{explanation} Representative compounds: {names}. "
            f"{loadable} of the top {len(top_results)} include loadable SMILES strings for direct use in PharmaOS."
        )
        return ChainOfThought(
            level=ThinkingLevel.RAG.value,
            confidence=0.9,
            reasoning="Resolved the query through the validated compound lookup service.",
            recommendation=recommendation,
            formula="Validated Compound Resolver",
        )

    def _rag_search(self, query: str) -> Optional[ChainOfThought]:
        if not SKLEARN_AVAILABLE or self.vectorizer is None:
            best_match = None
            max_hits = 0
            q_words = set(query.lower().split())
            for entry in self.corpus:
                hits = sum(1 for keyword in entry["keywords"] if keyword in q_words)
                if hits > max_hits:
                    max_hits = hits
                    best_match = entry

            if max_hits > 0 and best_match:
                return ChainOfThought(
                    level=ThinkingLevel.RAG.value,
                    confidence=min(0.5 + (max_hits * 0.1), 0.9),
                    reasoning=f"Matched {max_hits} corpus keywords.",
                    recommendation=best_match["knowledge"],
                    formula=best_match["formula"],
                )
            return None

        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        best_idx = int(similarities.argmax())
        best_score = float(similarities[best_idx])

        if best_score > 0.2:
            entry = self.corpus[best_idx]
            return ChainOfThought(
                level=ThinkingLevel.RAG.value,
                confidence=best_score,
                reasoning=f"Retrieved closest medicinal-chemistry note with cosine similarity {best_score:.2f}.",
                recommendation=entry["knowledge"],
                formula=entry["formula"],
            )
        return None

    def _llm_oracle(self, query: str, context: str) -> Optional[ChainOfThought]:
        if not self.llm_backends:
            return None

        prompt = (
            "You are PharmaAgent, a concise chemistry, pharmacology, and drug-discovery copilot.\n"
            "Respond with practical, trustworthy guidance grounded in chemistry, medicinal chemistry, "
            "and when relevant Lipinski, QED, SA, and ADMET.\n"
            "Use plain, human language and explain tradeoffs clearly without sounding robotic.\n\n"
            f"Context:\n{context}\n\n"
            f"User query:\n{query}"
        )

        for backend in self.llm_backends:
            try:
                response = backend["client"].chat.completions.create(
                    model=backend["model"],
                    temperature=0.2,
                    max_tokens=180,
                    timeout=6.0,
                    messages=[
                        {"role": "system", "content": "You are a precise medicinal chemistry assistant."},
                        {"role": "system", "content": "When a common product name is not a single molecule, explain that clearly and offer representative compounds."},
                        {"role": "user", "content": prompt},
                    ],
                )
                text = (response.choices[0].message.content or "").strip()
                if text:
                    return ChainOfThought(
                        level=ThinkingLevel.LLM.value,
                        confidence=0.85,
                        reasoning=(
                            "LLM Oracle returned a model-backed chemistry suggestion via the "
                            f"{backend['provider']} backend."
                        ),
                        recommendation=text,
                        formula="LLM Generative Priority",
                        provider=backend["provider"],
                    )
            except Exception:
                continue
        return None

    def get_reasoning_trace(self, query: str, context: str = "") -> ChainOfThought:
        cot = self._fast_cache_match(query)
        if cot:
            return cot

        compound_cot = self._compound_lookup_match(query)
        if compound_cot:
            return compound_cot

        cot = self._rag_search(query)
        if cot and cot.confidence > 0.8:
            return cot

        llm_cot = self._llm_oracle(query, context)
        if llm_cot:
            return llm_cot

        if cot:
            return cot

        return ChainOfThought(
            level=ThinkingLevel.RAG.value,
            confidence=0.1,
            reasoning="No strong match found, falling back to generic medicinal-chemistry advice.",
            recommendation=(
                "Try asking about a concrete chemical, drug, product class, or property such as "
                "detergent surfactants, aspirin, solubility, Lipinski violations, or hERG risk so "
                "the guidance can be specific."
            ),
            formula="Fallback Heuristic",
        )

    def generate_response(self, query: str, context: str = "") -> str:
        trace = self.get_reasoning_trace(query, context)
        prefix = f"[Layer: {trace.level}] "
        if trace.formula and trace.formula not in {"Fallback Heuristic", "SMILES Grammar", "LLM Generative Priority"}:
            prefix += f"[Rule: {trace.formula}] "
        return f"{prefix}\n\n{trace.recommendation}"


agent = PharmaAgent()

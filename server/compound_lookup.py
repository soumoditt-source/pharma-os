from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import requests
from openai import OpenAI


PUBCHEM_PROPERTY_FIELDS = (
    "Title,IUPACName,MolecularFormula,MolecularWeight,"
    "SMILES,ConnectivitySMILES,CanonicalSMILES,InChIKey"
)
PUBCHEM_TIMEOUT = 12

COMMON_QUERY_MAP: Dict[str, Dict[str, Any]] = {
    "detergent": {
        "summary": (
            "Detergent is a product class rather than one single molecule. "
            "Representative detergent surfactants are shown below."
        ),
        "candidates": [
            "sodium dodecyl sulfate",
            "sodium laureth sulfate",
            "sodium lauryl sulfate",
            "linear alkylbenzene sulfonate",
        ],
    },
    "soap": {
        "summary": "Soap is a mixture class. Representative soap molecules are fatty-acid salts.",
        "candidates": [
            "sodium stearate",
            "sodium palmitate",
            "potassium oleate",
        ],
    },
    "bleach": {
        "summary": "Bleach is a household category. Common active chemicals are shown below.",
        "candidates": [
            "sodium hypochlorite",
            "hydrogen peroxide",
        ],
    },
    "salt": {
        "summary": "Table salt usually refers to sodium chloride.",
        "candidates": ["sodium chloride"],
    },
    "sugar": {
        "summary": "Sugar can mean several carbohydrates. Common examples are shown below.",
        "candidates": ["sucrose", "glucose", "fructose"],
    },
    "vinegar": {
        "summary": "The defining acidic component of vinegar is acetic acid.",
        "candidates": ["acetic acid"],
    },
    "baking soda": {
        "summary": "Baking soda is sodium bicarbonate.",
        "candidates": ["sodium bicarbonate"],
    },
    "rubbing alcohol": {
        "summary": "Rubbing alcohol usually refers to isopropyl alcohol.",
        "candidates": ["isopropyl alcohol"],
    },
    "alcohol": {
        "summary": "Alcohol is a chemical family. Common everyday examples are shown below.",
        "candidates": ["ethanol", "isopropyl alcohol", "methanol"],
    },
    "perfume": {
        "summary": "Perfume is a formulation class. Representative aromatic compounds are shown below.",
        "candidates": ["linalool", "limonene", "vanillin"],
    },
    "sunscreen": {
        "summary": "Sunscreen is a formulation class with multiple UV filters.",
        "candidates": ["avobenzone", "oxybenzone", "octisalate"],
    },
    "detergent surfactant": {
        "summary": "Representative detergent surfactants are shown below.",
        "candidates": [
            "sodium dodecyl sulfate",
            "sodium laureth sulfate",
            "cetyltrimethylammonium bromide",
        ],
    },
}

QUERY_ALIASES = {
    "paracetamol": "acetaminophen",
    "tylenol": "acetaminophen",
    "advil": "ibuprofen",
    "motrin": "ibuprofen",
    "panadol": "acetaminophen",
}


def _normalize_query(query: str) -> str:
    return " ".join((query or "").strip().lower().split())


def _dedupe_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    deduped: List[Dict[str, Any]] = []
    for result in results:
        key = result.get("smiles") or result.get("inchikey") or result.get("name")
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(result)
    return deduped


def _property_to_result(prop: Dict[str, Any], source: str) -> Dict[str, Any]:
    smiles = (
        prop.get("SMILES")
        or prop.get("ConnectivitySMILES")
        or prop.get("CanonicalSMILES")
        or ""
    )
    return {
        "name": prop.get("Title") or prop.get("IUPACName") or "Unknown compound",
        "iupac_name": prop.get("IUPACName") or prop.get("Title") or "Unknown compound",
        "formula": prop.get("MolecularFormula") or "",
        "molecular_weight": prop.get("MolecularWeight"),
        "smiles": smiles,
        "inchikey": prop.get("InChIKey") or "",
        "cid": prop.get("CID"),
        "source": source,
        "loadable": bool(smiles),
    }


def _fetch_pubchem_properties(name: str) -> List[Dict[str, Any]]:
    response = requests.get(
        "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
        f"{requests.utils.quote(name)}/property/{PUBCHEM_PROPERTY_FIELDS}/JSON",
        timeout=PUBCHEM_TIMEOUT,
    )
    response.raise_for_status()
    payload = response.json()
    properties = payload.get("PropertyTable", {}).get("Properties", [])
    return [_property_to_result(prop, "pubchem_exact") for prop in properties]


def _autocomplete_pubchem(query: str, limit: int = 8) -> List[str]:
    response = requests.get(
        "https://pubchem.ncbi.nlm.nih.gov/rest/autocomplete/compound/"
        f"{requests.utils.quote(query)}/JSON?limit={limit}",
        timeout=PUBCHEM_TIMEOUT,
    )
    response.raise_for_status()
    payload = response.json()
    return payload.get("dictionary_terms", {}).get("compound", []) or []


def _llm_candidate_names(query: str) -> List[str]:
    token = os.getenv("HF_TOKEN")
    if not token:
        return []

    client = OpenAI(
        base_url=os.getenv("API_BASE_URL", "https://router.huggingface.co/v1"),
        api_key=token,
    )
    model_name = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
    prompt = (
        "You help resolve human search phrases into real chemical compound names.\n"
        "Return strict JSON with keys intent, explanation, and candidates.\n"
        "Candidates must be 1 to 5 real compound names likely to exist in PubChem.\n"
        "If the query is a product class like detergent or perfume, return representative compounds.\n"
        f"Query: {query}"
    )

    try:
        response = client.chat.completions.create(
            model=model_name,
            temperature=0,
            max_tokens=200,
            messages=[
                {"role": "system", "content": "Return only valid JSON."},
                {"role": "user", "content": prompt},
            ],
        )
        content = (response.choices[0].message.content or "").strip()
        parsed = json.loads(content)
        candidates = parsed.get("candidates", [])
        if isinstance(candidates, list):
            return [str(item).strip() for item in candidates if str(item).strip()]
    except Exception:
        return []
    return []


def resolve_compound_query(query: str, limit: int = 8) -> Dict[str, Any]:
    normalized = _normalize_query(query)
    if not normalized:
        return {
            "query": query,
            "normalized_query": normalized,
            "summary": "Enter a compound name, common product name, or chemical category.",
            "results": [],
            "sources": [],
        }

    results: List[Dict[str, Any]] = []
    sources: List[str] = []
    summary = ""

    mapped_query = QUERY_ALIASES.get(normalized, normalized)
    concept_entry = COMMON_QUERY_MAP.get(mapped_query)
    if concept_entry:
        summary = concept_entry["summary"]
        sources.append("curated_common_name_map")
        for candidate in concept_entry["candidates"]:
            try:
                results.extend(_fetch_pubchem_properties(candidate))
            except requests.RequestException:
                continue

    if not results:
        try:
            exact = _fetch_pubchem_properties(mapped_query)
            if exact:
                sources.append("pubchem_exact")
                results.extend(exact)
        except requests.RequestException:
            pass

    if not results:
        try:
            suggestions = _autocomplete_pubchem(mapped_query, limit=limit)
            if suggestions:
                sources.append("pubchem_autocomplete")
            for suggestion in suggestions:
                try:
                    results.extend(_fetch_pubchem_properties(suggestion)[:1])
                except requests.RequestException:
                    continue
                if len(results) >= limit:
                    break
        except requests.RequestException:
            pass

    if not results:
        for suggestion in _llm_candidate_names(mapped_query):
            try:
                results.extend(_fetch_pubchem_properties(suggestion)[:1])
                sources.append("openai_validated_fallback")
            except requests.RequestException:
                continue
            if len(results) >= limit:
                break

    deduped = _dedupe_results(results)[:limit]
    if not summary:
        if deduped:
            loadable_count = sum(1 for result in deduped if result.get("loadable"))
            summary = (
                f"Found {len(deduped)} validated PubChem match"
                f"{'' if len(deduped) == 1 else 'es'} for '{query}'. "
                f"{loadable_count} contain loadable SMILES strings."
            )
        else:
            summary = (
                "No validated compound match was found. Try a more specific chemical name, "
                "brand-generic synonym, or a product category like detergent, bleach, or sunscreen."
            )

    return {
        "query": query,
        "normalized_query": mapped_query,
        "summary": summary,
        "results": deduped,
        "sources": sources,
    }

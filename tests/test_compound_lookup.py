from __future__ import annotations

from fastapi.testclient import TestClient

from server.app import app
from server import compound_lookup


def test_resolve_compound_query_handles_common_product_name(monkeypatch):
    def fake_fetch(name: str):
        return [
            {
                "name": name.title(),
                "iupac_name": name.title(),
                "formula": "C12H25SO4Na",
                "molecular_weight": 288.38,
                "smiles": "CCCCCCCCCCCCOS(=O)(=O)[O-].[Na+]",
                "inchikey": name,
                "cid": 1,
                "source": "fake_pubchem",
                "loadable": True,
            }
        ]

    monkeypatch.setattr(compound_lookup, "_fetch_pubchem_properties", fake_fetch)

    payload = compound_lookup.resolve_compound_query("detergent")

    assert payload["results"]
    assert "product class" in payload["summary"].lower()
    assert any(result["loadable"] for result in payload["results"])


def test_compound_lookup_endpoint_returns_backend_results(monkeypatch):
    monkeypatch.setattr(
        "server.app.resolve_compound_query",
        lambda query: {
            "query": query,
            "normalized_query": query,
            "summary": "Found validated results.",
            "results": [
                {
                    "name": "Acetaminophen",
                    "iupac_name": "N-(4-hydroxyphenyl)acetamide",
                    "formula": "C8H9NO2",
                    "molecular_weight": 151.16,
                    "smiles": "CC(=O)Nc1ccc(cc1)O",
                    "inchikey": "RZVAJINKPMORJF-UHFFFAOYSA-N",
                    "cid": 1983,
                    "source": "test",
                    "loadable": True,
                }
            ],
            "sources": ["test"],
        },
    )

    client = TestClient(app)
    response = client.get("/api/compound_lookup", params={"q": "paracetamol"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["results"][0]["smiles"] == "CC(=O)Nc1ccc(cc1)O"


def test_property_to_result_prefers_pubchem_smiles_field():
    result = compound_lookup._property_to_result(
        {
            "Title": "Acetaminophen",
            "IUPACName": "N-(4-hydroxyphenyl)acetamide",
            "MolecularFormula": "C8H9NO2",
            "MolecularWeight": 151.16,
            "SMILES": "CC(=O)NC1=CC=C(O)C=C1",
            "InChIKey": "RZVAJINKPMORJF-UHFFFAOYSA-N",
            "CID": 1983,
        },
        "test",
    )

    assert result["smiles"] == "CC(=O)NC1=CC=C(O)C=C1"
    assert result["loadable"] is True

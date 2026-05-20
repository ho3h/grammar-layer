"""Smoke tests for the fingerprint API.

Doesn't load models — just verifies the data lookups against cached reports.
"""

from __future__ import annotations

import pytest

from neograph.fingerprint import (
    CANONICAL_FINGERPRINT,
    cross_model_routing,
    identify_copula_opposers,
    known_fingerprint_pairs,
)


def test_canonical_fingerprint_has_known_features():
    """The canonical fingerprint table includes the documented Gemma + Pythia features."""
    gemma_feats = {f["feature"] for f in CANONICAL_FINGERPRINT["gemma"]}
    assert 15596 in gemma_feats, "Gemma 2 2B fingerprint must include f15596"
    assert 10142 in gemma_feats, "Gemma 2 2B fingerprint must include f10142"

    pythia_feats = {f["feature"] for f in CANONICAL_FINGERPRINT["pythia_70m"]}
    assert 23527 in pythia_feats, "Pythia 70M fingerprint must include f23527"

    # GPT-2 has no grammar fingerprint
    assert CANONICAL_FINGERPRINT["gpt2"] == []


def test_identify_returns_feature_records():
    """identify_copula_opposers returns typed FeatureRecord objects."""
    records = identify_copula_opposers("gemma")
    assert len(records) >= 2
    for r in records:
        assert r.feature >= 0
        assert r.label
        assert r.role == "opposer"


def test_cross_model_routing_capital_jp():
    """cross_model_routing on the canonical capital-jp prompt returns at least Gemma + GPT-2."""
    routing = cross_model_routing("capital-jp")
    assert "gemma" in routing, "capital-jp should appear in Gemma's load-bearing report"
    assert "gpt2" in routing, "capital-jp should appear in GPT-2's load-bearing report"

    gemma_blob = routing["gemma"]
    assert "supporting" in gemma_blob and len(gemma_blob["supporting"]) > 0
    assert "opposing" in gemma_blob and len(gemma_blob["opposing"]) > 0
    # The known fingerprint feature should appear in Gemma's opposing top-K
    gemma_opposers = {e["feature_index"] for e in gemma_blob["opposing"]}
    assert 15596 in gemma_opposers, "f15596 must oppose capital-jp in Gemma"


def test_known_pairs_flat_listing():
    """known_fingerprint_pairs returns a flat (model, feature, label) list."""
    pairs = known_fingerprint_pairs()
    assert len(pairs) >= 4  # At least 2 Gemma + 1 Pythia + 1+ Gemma 1
    models_with_fingerprint = {p[0] for p in pairs}
    assert "gemma" in models_with_fingerprint
    assert "pythia_70m" in models_with_fingerprint
    assert "gpt2" not in models_with_fingerprint  # No grammar fingerprint in GPT-2


def test_unknown_model_returns_empty():
    """Unknown model nickname returns an empty list (not an error)."""
    assert identify_copula_opposers("mythical-model") == []

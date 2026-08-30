# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused Studio model catalogue contracts

"""Focused descriptor-backed model catalogue contracts."""

from .studio_model_catalogue_support import *


def test_list_models_serves_declared_family_and_provenance() -> None:
    summary = _adex_summary()
    assert summary["family"] == "Integrate-and-Fire"
    assert summary["category"] == "Integrate-and-Fire"
    assert summary["category_slug"] == "integrate-and-fire"
    assert summary["category_source"] == "declared"
    provenance = cast(dict[str, object], summary["provenance"])
    assert provenance["doi"] == "10.1152/jn.00686.2005"


def test_summary_exposes_completeness_tier_and_evidence_kind() -> None:
    by_name = {m["name"]: m for m in list_models()}
    adex = by_name["AdExNeuron"]
    # AdEx is engineering-verified (python+rust + golden trace) → kernel tier 3.
    assert adex["tier"] == 3
    assert adex["evidence_kind"] == "measured"
    fhn = by_name["FitzHughNagumoNeuron"]
    assert fhn["tier"] == 3
    assert fhn["evidence_kind"] == "measured"
    lapicque = by_name["LapicqueNeuron"]
    # Lapicque now carries python+rust backends with a golden trace → kernel tier 3.
    assert lapicque["tier"] == 3
    assert lapicque["evidence_kind"] == "measured"


def test_summary_exposes_dual_readiness_axes() -> None:
    """The catalogue list carries both the science (S0-S5) and silicon axes."""
    adex = _adex_summary()
    # The full science axis never sits below the legacy kernel tier.
    assert cast(int, adex["science_tier"]) >= cast(int, adex["tier"])
    assert adex["science_label"] == f"S{adex['science_tier']}"
    # S5: S3 kernel + dynamics_faithful + class-validated parity evidence.
    assert adex["science_tier"] == 5
    assert adex["science_label"] == "S5"
    # AdEx has schema→RTL cosim plus a checked coarse-synthesis report.
    assert adex["silicon_tier"] == 2
    assert adex["silicon_label"] == "H2"


def test_summary_exposes_studio_consumption_contract() -> None:
    """Every browse entry carries the descriptor facts required by Studio."""
    models = list_models()
    required = {
        "validation_metric",
        "integration_method",
        "terminal_silicon_tier",
        "terminal_reason",
    }
    assert all(required <= set(model) for model in models)

    adex = _adex_summary()
    assert adex["validation_metric"] == "parity"
    assert adex["integration_method"] == "euler"
    assert adex["terminal_silicon_tier"] == "H2"
    assert adex["terminal_reason"] == (
        "Q16.16 co-simulation, coarse synthesis, and bounded reset/event formal safety are "
        "complete; timing, PPA, board, device, physical silicon, and universal formal "
        "equivalence remain open."
    )


def test_introspected_summary_defaults_to_below_readiness() -> None:
    """A descriptor-less catalogue entry reports S0 and no silicon tier."""
    summary = _introspected_summary("AdExNeuron")
    assert summary["science_tier"] == 0
    assert summary["science_label"] == "S0"
    assert summary["silicon_tier"] is None
    assert summary["silicon_label"] == "none"
    assert summary["validation_metric"] == "none"
    assert summary["integration_method"] == "unknown"
    assert summary["terminal_silicon_tier"] == ""
    assert "Descriptor unavailable" in cast(str, summary["terminal_reason"])


def test_detail_readiness_block_is_auditable() -> None:
    """Model detail exposes the dual-axis readiness with its evidence facets."""
    detail = get_model_detail("AdExNeuron")
    assert detail is not None
    readiness = cast(dict[str, object], detail["readiness"])
    assert readiness["science_label"] == f"S{readiness['science_tier']}"
    assert readiness["science_tier"] == 5
    assert readiness["silicon_tier"] == 2
    assert readiness["silicon_label"] == "H2"
    # Perfect: S5 plus declared terminal H-tier (H2) is met via synth evidence.
    assert readiness["is_perfect"] is True
    validation = cast(dict[str, object], readiness["validation"])
    assert validation["metric"] == "parity"
    assert validation["dynamics_faithful"] is True
    silicon = cast(dict[str, object], readiness["silicon"])
    assert silicon["compiles"] is True
    assert silicon["cosim_validated"] is True
    assert silicon["clock_mhz"] is None


def test_no_model_falls_into_an_other_bucket() -> None:
    """Every model now declares a real family — the 'Other' bucket is gone."""

    categories = {str(m["category"]) for m in list_models()}
    assert "Other" not in categories
    assert all(m["category_source"] == "declared" for m in list_models())


def test_get_model_detail_serves_descriptor_parameters_and_dynamics() -> None:
    detail = get_model_detail("AdExNeuron")
    assert detail is not None
    param = detail["params"][0]
    assert set(param) >= {"name", "default", "unit", "range", "meaning"}
    assert "v" in detail["dynamics"]
    assert any(b["name"] == "python" for b in detail["backends"])
    assert detail["family"] == "Integrate-and-Fire"


def test_model_detail_exposes_measured_golden_trace_digest_variants() -> None:
    """Studio clients can verify every typed platform-compatible trace digest."""
    detail = get_model_detail("ChialvoMapNeuron")
    assert detail is not None
    reproducibility = cast(dict[str, object], detail["reproducibility"])
    assert reproducibility["golden_trace_sha256"] == (
        "ccd95d8035517d0684efbc35c60a1e0769954a77f54b77b3aecdf5d8d5352ff4"
    )
    assert reproducibility["golden_trace_sha256_variants"] == []


def test_model_facets_cover_the_whole_catalogue() -> None:
    facets = model_facets()
    assert facets["total"] == len(list_models())
    assert sum(f["count"] for f in facets["families"]) == facets["total"]
    families = {f["family"] for f in facets["families"]}
    assert "Cerebellar" in families
    assert "Integrate-and-Fire" in families
    # The measured behaviour facet is present and ordered most-common first.
    behaviors = facets["behaviors"]
    assert all({"tag", "count"} <= set(entry) for entry in behaviors)
    counts = [entry["count"] for entry in behaviors]
    assert counts == sorted(counts, reverse=True)
    # Dual-axis discovery counts cover the whole catalogue exactly once each.
    science_tiers = cast(dict[str, int], facets["science_tiers"])
    silicon_tiers = cast(dict[str, int], facets["silicon_tiers"])
    assert sum(science_tiers.values()) == facets["total"]
    assert sum(silicon_tiers.values()) == facets["total"]
    assert all(label.startswith("S") for label in science_tiers)
    assert "none" in silicon_tiers or any(label.startswith("H") for label in silicon_tiers)


def test_introspected_fallback_flags_inferred_category() -> None:
    """A model without a descriptor falls back to an inferred category."""

    summary = _introspected_summary("AdExNeuron")
    assert summary["category_source"] == "inferred"
    assert summary["name"] == "AdExNeuron"
    assert summary["n_params"] >= 1

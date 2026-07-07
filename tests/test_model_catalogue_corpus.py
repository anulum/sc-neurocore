# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Model descriptor corpus gate

"""Gate for the on-disk model descriptor corpus.

Guarantees that every registered model has a committed descriptor, that the
descriptor's structural fields (which parameters and state variables exist and
their defaults) stay in sync with the model code, and that the curation merge
preserves human-curated content. The descriptor corpus cannot silently drift
from the implementation.
"""

from __future__ import annotations

import pytest

from sc_neurocore.neurons.descriptor_generator import (
    generate_descriptor,
    generate_descriptor_payload,
    merge_descriptor_payloads,
)
from sc_neurocore.neurons import model_catalogue as model_catalogue_module
from sc_neurocore.neurons.model_catalogue import (
    CatalogueCoverage,
    catalogue_descriptor_coverage,
    descriptor_path,
    load_descriptor,
    load_descriptor_payload,
)
from sc_neurocore.neurons.models import _CLASS_TO_MODULE


def test_every_model_has_a_committed_descriptor() -> None:
    """Every registered model has a committed, schema-valid descriptor."""

    missing = sorted(name for name in _CLASS_TO_MODULE if load_descriptor(name) is None)
    assert missing == [], f"models without a committed descriptor: {missing}"


@pytest.mark.parametrize("class_name", sorted(_CLASS_TO_MODULE))
def test_committed_descriptor_matches_code(class_name: str) -> None:
    """Committed structural fields stay in sync with the model code.

    Parameter names and defaults, state names and initial values, and the
    timestep must match what the generator reads from the code; otherwise the
    descriptor has drifted from the implementation and must be regenerated.
    """

    committed = load_descriptor(class_name)
    assert committed is not None
    generated = generate_descriptor(class_name)

    assert {p.name: p.default for p in committed.parameters} == {
        p.name: p.default for p in generated.parameters
    }
    assert {s.name: s.init for s in committed.state} == {s.name: s.init for s in generated.state}
    assert committed.dt == generated.dt


def test_coverage_describes_every_model() -> None:
    """Descriptor coverage accounts for the whole registered catalogue."""

    coverage = catalogue_descriptor_coverage()
    assert coverage.total_models == len(_CLASS_TO_MODULE)
    assert coverage.described == len(_CLASS_TO_MODULE)
    assert sum(coverage.tier_counts.values()) == coverage.described
    # Every model carries a declared family + category, so none remains at tier 0;
    # all described models are at least Tier 1 (some curated higher).
    assert coverage.tier_counts[0] == 0
    assert coverage.tier_counts[1] + coverage.tier_counts[2] + coverage.tier_counts[3] == (
        coverage.described
    )


def test_coverage_reports_both_readiness_axes_consistently() -> None:
    """Science (S0-S5) and silicon (H0-H5) axes cover every described model and
    agree with the legacy kernel on S0-S3."""

    coverage = catalogue_descriptor_coverage()

    # Both axes are total over the described corpus.
    assert set(coverage.science_tier_counts) == {0, 1, 2, 3, 4, 5}
    assert set(coverage.silicon_tier_counts) == {"none", "H0", "H1", "H2", "H3", "H4", "H5"}
    assert sum(coverage.science_tier_counts.values()) == coverage.described
    assert sum(coverage.silicon_tier_counts.values()) == coverage.described

    # The legacy kernel is exactly the science axis truncated at S3:
    # descriptor_completeness_tier(d) == min(science_tier(d), 3).
    assert coverage.tier_counts[0] == coverage.science_tier_counts[0]
    assert coverage.tier_counts[1] == coverage.science_tier_counts[1]
    assert coverage.tier_counts[2] == coverage.science_tier_counts[2]
    assert coverage.tier_counts[3] == (
        coverage.science_tier_counts[3]
        + coverage.science_tier_counts[4]
        + coverage.science_tier_counts[5]
    )


def test_descriptor_access_rejects_path_like_class_names() -> None:
    """Descriptor lookup fails closed before path-like names reach the filesystem."""

    invalid_names = ("../AdExNeuron", "nested/AdExNeuron", "AdExNeuron.toml", "")

    for class_name in invalid_names:
        with pytest.raises(ValueError, match="model class name"):
            descriptor_path(class_name)
        with pytest.raises(ValueError, match="model class name"):
            load_descriptor_payload(class_name)
        with pytest.raises(ValueError, match="model class name"):
            load_descriptor(class_name)


def test_missing_descriptor_branch_and_public_summary_are_stable() -> None:
    """Valid absent descriptors return ``None`` and summaries stay JSON-safe."""

    assert load_descriptor_payload("DescriptorAbsentForCoverage") is None
    assert load_descriptor("DescriptorAbsentForCoverage") is None

    coverage = CatalogueCoverage(
        total_models=3,
        described=2,
        tier_counts={3: 1, 1: 1, 0: 0, 2: 0},
        science_tier_counts={5: 0, 4: 0, 3: 1, 2: 0, 1: 1, 0: 0},
        silicon_tier_counts={"none": 2, "H0": 0, "H1": 0, "H2": 0, "H3": 0, "H4": 0, "H5": 0},
        citeable=1,
        fully_curated_parameters=1,
    )

    assert coverage.to_public_dict() == {
        "total_models": 3,
        "described": 2,
        "undescribed": 1,
        "tier_counts": {0: 0, 1: 1, 2: 0, 3: 1},
        "science_tier_counts": {0: 0, 1: 1, 2: 0, 3: 1, 4: 0, 5: 0},
        "silicon_tier_counts": {"none": 2, "H0": 0, "H1": 0, "H2": 0, "H3": 0, "H4": 0, "H5": 0},
        "citeable": 1,
        "fully_curated_parameters": 1,
    }


def test_catalogue_coverage_counts_valid_absent_descriptors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The aggregate coverage report counts registered models without descriptors."""

    monkeypatch.setattr(
        model_catalogue_module,
        "_CLASS_TO_MODULE",
        {"AdExNeuron": "adex", "DescriptorAbsentForCoverage": "missing"},
    )

    coverage = catalogue_descriptor_coverage()

    assert coverage.total_models == 2
    assert coverage.described == 1
    assert sum(coverage.tier_counts.values()) == 1
    assert coverage.to_public_dict()["undescribed"] == 1


def test_every_parameter_has_a_complete_schema() -> None:
    """Every catalogued parameter documents its unit, valid range and meaning.

    This is independent of the citation tier: a model may stay Tier 1 for want of
    a confirmable DOI yet still describe each of its parameters fully. The gate
    keeps that completeness from regressing once filled.
    """

    incomplete: list[str] = []
    for class_name in _CLASS_TO_MODULE:
        descriptor = load_descriptor(class_name)
        assert descriptor is not None
        for parameter in descriptor.parameters:
            if not parameter.unit or parameter.value_range is None or not parameter.meaning:
                incomplete.append(f"{class_name}.{parameter.name}")
    assert not incomplete, (
        f"{len(incomplete)} parameters miss unit/range/meaning: {sorted(incomplete)[:20]}"
    )


def test_every_parameter_default_lies_within_its_range() -> None:
    """A declared valid range must contain the parameter's own default value.

    A default outside its range is a curation bug — it makes the documented
    bounds contradict the shipped model. This gate catches both a too-narrow
    range and a unit mismatch (e.g. a normalised default under millivolt bounds).
    """

    out_of_range: list[str] = []
    for class_name in _CLASS_TO_MODULE:
        descriptor = load_descriptor(class_name)
        assert descriptor is not None
        for parameter in descriptor.parameters:
            if parameter.value_range is None:
                continue
            low, high = parameter.value_range
            if not low <= parameter.default <= high:
                out_of_range.append(
                    f"{class_name}.{parameter.name}={parameter.default} not in [{low}, {high}]"
                )
    assert not out_of_range, (
        f"{len(out_of_range)} defaults outside their range: {sorted(out_of_range)[:20]}"
    )


def test_merge_preserves_curation_and_follows_code() -> None:
    """A merge keeps human curation but takes structure from the code."""

    regenerated = generate_descriptor_payload("AdExNeuron")
    curated = {
        "metadata": {"family": "Integrate-and-Fire", "category": "adaptive-exponential"},
        "parameters": {
            "tau": {"unit": "ms", "range": [1.0, 100.0], "meaning": "membrane time constant"},
            # A parameter the code no longer has must be dropped by the merge.
            "obsolete_param": {"unit": "mV", "meaning": "removed"},
        },
        "documentation": {"notes": "curator note"},
    }

    merged = merge_descriptor_payloads(curated, regenerated)

    # Curation preserved.
    assert merged["metadata"]["family"] == "Integrate-and-Fire"
    assert merged["metadata"]["category"] == "adaptive-exponential"
    assert merged["parameters"]["tau"]["unit"] == "ms"
    assert merged["parameters"]["tau"]["meaning"] == "membrane time constant"
    assert merged["documentation"]["notes"] == "curator note"
    # Structure follows the code: defaults from the model, no obsolete params.
    assert merged["parameters"]["tau"]["default"] == regenerated["parameters"]["tau"]["default"]
    assert "obsolete_param" not in merged["parameters"]

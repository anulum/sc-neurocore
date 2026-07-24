# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (parse_and_rejects) from former test_model_descriptor.py

from __future__ import annotations

from tests.model_descriptor_support import *  # noqa: F403


def test_parse_minimal_descriptor() -> None:
    """A minimal valid payload parses into a Tier-0 descriptor."""

    descriptor = parse_model_descriptor(_minimal_payload())
    assert descriptor.class_name == "AdExNeuron"
    assert descriptor.schema_version == MODEL_DESCRIPTOR_SCHEMA_VERSION
    assert descriptor.display_name == "AdEx"
    assert [s.name for s in descriptor.state] == ["v"]
    assert [p.name for p in descriptor.parameters] == ["tau"]
    assert descriptor_completeness_tier(descriptor) == 0


def test_parse_reads_doi_is_translation_flag() -> None:
    """A provenance section can mark its DOI as a translation of the cited work."""
    payload = _minimal_payload()
    payload["provenance"] = {
        "authors": ["Lapicque, L."],
        "year": 1907,
        "doi": "10.1007/s00422-007-0189-6",
        "doi_is_translation": True,
    }
    descriptor = parse_model_descriptor(payload)
    assert descriptor.provenance.doi_is_translation is True
    # The flag defaults to False when absent, so ordinary citations are unaffected.
    payload["provenance"].pop("doi_is_translation")
    assert parse_model_descriptor(payload).provenance.doi_is_translation is False


def test_parse_rejects_missing_class_name() -> None:
    payload = _minimal_payload()
    del payload["metadata"]["class_name"]
    with pytest.raises(ModelDescriptorError, match="class_name"):
        parse_model_descriptor(payload)


def test_parse_rejects_unknown_schema_version() -> None:
    payload = _minimal_payload()
    payload["metadata"]["schema_version"] = 1
    with pytest.raises(ModelDescriptorError, match="schema_version"):
        parse_model_descriptor(payload)


def test_parse_rejects_missing_required_metadata_section() -> None:
    """The metadata table is mandatory for every model descriptor."""

    payload = _minimal_payload()
    del payload["metadata"]

    with pytest.raises(ModelDescriptorError, match=r"missing the \[metadata\] section"):
        parse_model_descriptor(payload)


def test_parse_rejects_non_table_sections() -> None:
    """Section values must be TOML-style tables, not sequences or scalars."""

    payload = _minimal_payload()
    payload["state"] = ["v"]

    with pytest.raises(ModelDescriptorError, match=r"\[state\] must be a table"):
        parse_model_descriptor(payload)


def test_parse_rejects_string_tag_fields() -> None:
    """Discovery tag fields must be lists so scalar strings do not split silently."""

    payload = _minimal_payload()
    payload["metadata"]["intended_use"] = "simulation"

    with pytest.raises(ModelDescriptorError, match="tag fields"):
        parse_model_descriptor(payload)


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (
            lambda m: m["metadata"].__setitem__("biophysical_detail", "quantum"),
            "biophysical_detail",
        ),
        (lambda m: m["metadata"].__setitem__("maturity", "legendary"), "maturity"),
        (lambda m: m["metadata"].__setitem__("category", "Not A Slug"), "category"),
        (lambda m: m.__setitem__("provenance", {"doi": "not-a-doi"}), "DOI"),
        (
            lambda m: m.__setitem__("reproducibility", {"golden_trace_sha256": "xyz"}),
            "golden_trace",
        ),
    ],
)
def test_parse_rejects_invalid_controlled_fields(
    mutate: Callable[[dict[str, Any]], None],
    match: str,
) -> None:
    payload = _minimal_payload()
    mutate(payload)
    with pytest.raises(ModelDescriptorError, match=match):
        parse_model_descriptor(payload)


def test_parse_reads_measured_golden_trace_digest_variants() -> None:
    """Measured platform variants remain typed, ordered, and primary-prefixed."""
    primary = "a" * 64
    variant = "b" * 64
    payload = _minimal_payload()
    payload["reproducibility"] = {
        "reference_config": "golden/adex.json",
        "golden_trace_sha256": primary,
        "golden_trace_sha256_variants": [variant],
    }

    reproducibility = parse_model_descriptor(payload).reproducibility

    assert reproducibility.golden_trace_sha256_variants == (variant,)
    assert reproducibility.golden_trace_digests == (primary, variant)


@pytest.mark.parametrize(
    "variants",
    [
        "b" * 64,
        ["not-a-digest"],
        ["a" * 64],
        ["b" * 64, "b" * 64],
    ],
)
def test_parse_rejects_invalid_golden_trace_digest_variants(variants: object) -> None:
    """Variant digests must be a unique, non-primary lowercase SHA-256 list."""
    payload = _minimal_payload()
    payload["reproducibility"] = {
        "reference_config": "golden/adex.json",
        "golden_trace_sha256": "a" * 64,
        "golden_trace_sha256_variants": variants,
    }

    with pytest.raises(ModelDescriptorError, match="golden_trace_sha256_variants"):
        parse_model_descriptor(payload)


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda m: m.__setitem__("integration", {"dt": True}), "expected a number"),
        (
            lambda m: m.__setitem__(
                "parameters",
                {"tau": {"default": 20.0, "range": [1.0]}},
            ),
            "range",
        ),
        (lambda m: m.__setitem__("provenance", {"authors": ["Brette"], "year": True}), "year"),
    ],
)
def test_parse_rejects_invalid_shape_and_numeric_fields(
    mutate: Callable[[dict[str, Any]], None],
    match: str,
) -> None:
    """Malformed numeric and structured fields fail with descriptor errors."""

    payload = _minimal_payload()
    mutate(payload)

    with pytest.raises(ModelDescriptorError, match=match):
        parse_model_descriptor(payload)


def test_parse_accepts_legacy_scalar_and_mapping_forms() -> None:
    """Legacy scalar fields and compact backend forms normalize deterministically."""

    payload = _minimal_payload()
    payload["provenance"] = {"author": "Brette"}
    payload["state"] = {"v": -65.0}
    payload["parameters"] = {"tau": 20}
    payload["dynamics"] = {
        "v": {"expr": "(-v + input) / tau"},
        "w": {"expr": 3.0},
        "u": "du/dt",
    }
    payload["backends"] = {"python": "implemented"}

    descriptor = parse_model_descriptor(payload)

    assert descriptor.provenance.authors == ("Brette",)
    assert [(state.name, state.init) for state in descriptor.state] == [("v", -65.0)]
    assert [(parameter.name, parameter.default) for parameter in descriptor.parameters] == [
        ("tau", 20.0)
    ]
    assert descriptor.dynamics == {"v": "(-v + input) / tau", "w": "", "u": "du/dt"}
    assert [(backend.name, backend.status, backend.parity) for backend in descriptor.backends] == [
        ("python", "implemented", "n/a")
    ]

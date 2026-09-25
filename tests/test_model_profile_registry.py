# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Per-profile inventory and validator registry

"""The registry admits a profile only through its own executable validators."""

from __future__ import annotations

import pytest

from sc_neurocore.neurons.facet_receipts import FACET_BY_NAME, FACETS
from sc_neurocore.neurons.model_identity import identity_registry, schema_for_class
from sc_neurocore.neurons.profile_registry import (
    ProfileRow,
    method_table,
    profile_inventory,
    summarise_inventory,
    validator_registry,
)
from sc_neurocore.neurons.universal_dsl import list_bundled_schemas


@pytest.fixture(scope="module")
def rows() -> tuple[ProfileRow, ...]:
    """Build the inventory once per module."""
    return profile_inventory()


def test_inventory_covers_every_bound_profile_exactly_once(rows: tuple[ProfileRow, ...]) -> None:
    """One row per bound (class, stem); every bundled stem appears; no alias rows."""
    keys = [(row.class_name, row.stem) for row in rows]
    assert len(keys) == len(set(keys))
    assert {row.stem for row in rows} == set(list_bundled_schemas())
    expected = {
        (name, profile.stem)
        for name, identity in identity_registry().items()
        if identity.kind != "api-alias"
        for profile in identity.schema_profiles
    }
    assert set(keys) == expected
    for row in rows:
        assert row.canonical == (schema_for_class(row.class_name) == row.stem)
        assert row.readiness.profile == row.stem
        assert [entry.facet for entry in row.registry] == [spec.name for spec in FACETS]
    canonical_per_class = {
        name: sum(1 for row in rows if row.class_name == name and row.canonical)
        for name in {row.class_name for row in rows}
    }
    assert set(canonical_per_class.values()) == {1}


def test_no_cross_profile_promotion(rows: tuple[ProfileRow, ...]) -> None:
    """Lapicque's receipts credit the lapicque profile only; lif is blocked on its own merits."""
    by_key = {(row.class_name, row.stem): row for row in rows}
    lapicque = by_key[("LapicqueNeuron", "lapicque")]
    lif = by_key[("LapicqueNeuron", "lif")]
    assert lapicque.canonical and not lif.canonical
    for facet in ("dynamics_faithful", "class_validated", "cosim"):
        assert lapicque.entry(facet).receipt.endswith("__lapicque.json")
        assert lif.entry(facet).receipt == ""
        assert lif.entry(facet).status == "located"
        assert lif.entry(facet).executable_validators
        assert lif.entry(facet).admitting_validators == ()
        assert all(binding.scope == "class" for binding in lif.entry(facet).validators)
    assert lif.admission == "blocked"
    assert any("has no validator or receipt of its own" in r for r in lif.admission_reasons)
    assert lif.readiness.verified_science <= 3
    assert lif.readiness.verified_silicon is None
    assert lapicque.profile.numerical.method == "map"
    assert lif.profile.numerical.method == "euler"


def test_admission_requires_an_executable_validator_per_declared_facet(
    rows: tuple[ProfileRow, ...],
) -> None:
    """A flag or a report without a test blocks; descriptive records are not executable."""
    for row in rows:
        if not row.profile.is_executable:
            assert row.admission == "not-executable"
            assert row.admission_reasons == row.profile.lowering.limits
            continue
        expected_reasons = []
        for entry in row.registry:
            spec = FACET_BY_NAME[entry.facet]
            if not entry.declared or not spec.evidence_field or entry.status == "bound":
                continue
            if not entry.admitting_validators:
                expected_reasons.append(entry.facet)
        if expected_reasons:
            assert row.admission == "blocked", (row.class_name, row.stem)
            named = {reason.split()[2] for reason in row.admission_reasons}
            assert set(expected_reasons) <= named, (row.class_name, row.stem)
        else:
            assert row.admission == "admitted", (row.class_name, row.stem, row.admission_reasons)
        backends = {
            entry.facet.split(":", 1)[1]
            for entry in row.registry
            if entry.facet.startswith("backend:") and entry.declared and entry.status != "bound"
        }
        assert set(row.unvalidated_backends) == backends
    summary = summarise_inventory(rows)
    assert summary["admission"]["not-executable"] == 13
    assert summary["admission"].get("admitted", 0) >= 1
    assert "LapicqueNeuron" in summary["multi_profile_classes"]


def test_synthesis_reports_without_a_test_block_admission(rows: tuple[ProfileRow, ...]) -> None:
    """A Yosys report file is evidence but not a validator: the row says so explicitly."""
    lapicque = next(
        row for row in rows if (row.class_name, row.stem) == ("LapicqueNeuron", "lapicque")
    )
    synthesis = lapicque.entry("synthesis")
    assert synthesis.declared
    assert [binding.reference.kind for binding in synthesis.validators] == ["artifact-file"]
    assert synthesis.executable_validators == ()
    assert any(
        "declared facet synthesis has no executable validator" in r
        for r in lapicque.admission_reasons
    )


def test_descriptor_method_labels_are_joined_not_overwritten(rows: tuple[ProfileRow, ...]) -> None:
    """The descriptor keeps the hand class's numerics label; the row shows both."""
    by_key = {(row.class_name, row.stem): row for row in rows}
    hh = by_key[("HodgkinHuxleyNeuron", "hodgkin_huxley")]
    assert (hh.descriptor_method, hh.profile.numerical.method, hh.method_agreement) == (
        "baseline_euler",
        "rk4",
        "differs",
    )
    lapicque = by_key[("LapicqueNeuron", "lapicque")]
    assert lapicque.descriptor_method == "exact_constant_voltage_flow"
    assert lapicque.method_agreement == "differs"
    for name in (
        "AiharaMapNeuron",
        "NagumoSatoMapNeuron",
        "SCAdaptiveThresholdMapNeuron",
        "SCChaoticMapNeuron",
    ):
        row = next(item for item in rows if item.class_name == name)
        assert (row.descriptor_method, row.descriptor_dt, row.method_agreement) == (
            "map",
            1.0,
            "same",
        )


def test_registry_payload_and_method_table_are_serialisable(rows: tuple[ProfileRow, ...]) -> None:
    """Every row projects to JSON-compatible data with the documented keys."""
    import json

    registry = validator_registry(rows)
    assert set(registry) == {(row.class_name, row.stem) for row in rows}
    for row in rows:
        payload = row.to_public_dict()
        assert json.loads(json.dumps(payload)) == payload
        assert set(payload) >= {
            "class_name",
            "stem",
            "identity_kind",
            "canonical",
            "realisation_kind",
            "scientific",
            "numerical",
            "lowering",
            "descriptor",
            "verified",
            "declared",
            "admission",
            "admission_reasons",
            "unvalidated_backends",
            "registry",
        }
    assert [row["method"] for row in method_table()] == [
        "euler",
        "gauss_seidel",
        "rk4",
        "exp_euler",
        "map",
    ]


def test_an_unknown_facet_is_not_answered_by_another(rows: tuple[ProfileRow, ...]) -> None:
    with pytest.raises(KeyError, match="no-such-facet"):
        rows[0].entry("no-such-facet")


def test_a_schema_validation_block_without_evidence_names_no_validator() -> None:
    from sc_neurocore.neurons.profile_registry import _schema_validators
    from sc_neurocore.neurons.readiness import REPO_ROOT

    assert _schema_validators({"validation": {}}, REPO_ROOT) == ()
    assert _schema_validators({"validation": {"evidence": ""}}, REPO_ROOT) == ()


def test_a_class_without_a_descriptor_says_so_rather_than_agreeing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every bound class ships a descriptor, so its absence is produced by withholding it."""
    from sc_neurocore.neurons import profile_registry

    identity = identity_registry()["SCLapicqueLIFNeuron"]
    stem = identity.schema_profiles[0].stem
    monkeypatch.setattr(profile_registry, "load_descriptor_payload", lambda _name: None)
    row = profile_registry.profile_row(identity, stem)
    assert row.method_agreement == "no-descriptor"
    assert (row.descriptor_method, row.descriptor_dt) == ("", None)

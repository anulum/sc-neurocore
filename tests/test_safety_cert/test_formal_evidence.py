# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Safety Certification Generator Tests

"""Focused tests for formal evidence."""

from typing import Any

import pytest

from sc_neurocore.safety_cert.safety_cert import (
    FormalProofCertificate,
    FormalProperty,
    FormalPropertyGapDetector,
    ProofTestCoverage,
    PropertyGap,
    SILLevel,
)


def _unsafe(value: object) -> Any:
    """Return a deliberately invalid runtime value for boundary tests."""
    return value


class TestFormalProofCertificate:
    def _props(self) -> list[FormalProperty]:
        return [
            FormalProperty("P1", "sc_lif_neuron", "No overflow", "assert", "proven"),
            FormalProperty("P2", "sc_lif_neuron", "Reset works", "assert", "proven"),
            FormalProperty("P3", "sc_encoder", "Cover fire", "cover", "proven"),
            FormalProperty("P4", "sc_dense", "Weight range", "assert", "failed"),
        ]

    def test_proven_count(self) -> None:
        cert = FormalProofCertificate(properties=self._props())
        assert cert.proven_count == 3

    def test_proven_count_rejects_corrupted_internal_state(self) -> None:
        cert = FormalProofCertificate()
        cert.properties.append(_unsafe("bad"))
        with pytest.raises(ValueError, match="FormalProperty"):
            _ = cert.proven_count

    def test_proven_count_rejects_corrupted_property_status(self) -> None:
        cert = FormalProofCertificate()
        prop = FormalProperty("P1", "m", "d", "assert", "proven")
        prop.status = _unsafe("bad")
        cert.properties.append(prop)
        with pytest.raises(ValueError, match="statuses"):
            _ = cert.proven_count

    def test_pass_rate(self) -> None:
        cert = FormalProofCertificate(properties=self._props())
        assert abs(cert.pass_rate - 0.75) < 0.01

    def test_total_count_rejects_corrupted_internal_state(self) -> None:
        cert = FormalProofCertificate()
        cert.properties.append(_unsafe("bad"))
        with pytest.raises(ValueError, match="FormalProperty"):
            _ = cert.total_count

    def test_total_count_rejects_corrupted_property_id(self) -> None:
        cert = FormalProofCertificate()
        prop = FormalProperty("P1", "m", "d", "assert", "proven")
        prop.prop_id = _unsafe("")
        cert.properties.append(prop)
        with pytest.raises(ValueError, match="prop_id"):
            _ = cert.total_count

    def test_compute_hash(self) -> None:
        cert = FormalProofCertificate(properties=self._props())
        h = cert.compute_hash()
        assert len(h) == 32

    def test_add_property_rejects_invalid_contract(self) -> None:
        cert = FormalProofCertificate()
        with pytest.raises(ValueError, match="prop"):
            cert.add_property(_unsafe("bad"))

    def test_hash_deterministic(self) -> None:
        cert = FormalProofCertificate(properties=self._props())
        assert cert.compute_hash() == cert.compute_hash()

    def test_compute_hash_rejects_duplicate_property_ids(self) -> None:
        cert = FormalProofCertificate(
            properties=[
                FormalProperty("P1", "m1", "d1", "assert", "proven"),
                FormalProperty("P1", "m2", "d2", "assert", "proven"),
            ]
        )
        with pytest.raises(ValueError, match="duplicate"):
            cert.compute_hash()

    def test_compute_hash_rejects_corrupted_internal_state(self) -> None:
        cert = FormalProofCertificate()
        cert.properties.append(_unsafe("bad"))
        with pytest.raises(ValueError, match="FormalProperty"):
            cert.compute_hash()

    def test_compute_hash_rejects_corrupted_property_module(self) -> None:
        cert = FormalProofCertificate()
        prop = FormalProperty("P1", "m", "d", "assert", "proven")
        prop.module = _unsafe("")
        cert.properties.append(prop)
        with pytest.raises(ValueError, match="modules"):
            cert.compute_hash()

    def test_generate_report(self) -> None:
        cert = FormalProofCertificate(properties=self._props())
        report = cert.generate_report()
        assert "Formal Proof Certificate" in report
        assert "P1" in report

    def test_generate_report_rejects_corrupted_internal_state(self) -> None:
        cert = FormalProofCertificate()
        cert.properties.append(_unsafe("bad"))
        with pytest.raises(ValueError, match="FormalProperty"):
            cert.generate_report()

    def test_generate_report_rejects_corrupted_property_fields(self) -> None:
        cert = FormalProofCertificate()
        prop = FormalProperty("P1", "m", "d", "assert", "proven")
        prop.prop_id = _unsafe("")
        cert.properties.append(prop)
        with pytest.raises(ValueError, match="prop_id"):
            cert.generate_report()

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"generation_timestamp": None}, "generation_timestamp"),
            ({"tool_version": ""}, "tool_version"),
            ({"certificate_hash": None}, "certificate_hash"),
            ({"properties": ["not-prop"]}, "properties"),
        ],
    )
    def test_formal_proof_certificate_rejects_invalid_contracts(
        self, kwargs: Any, match: Any
    ) -> None:
        values = {
            "properties": self._props(),
            "generation_timestamp": "",
            "tool_version": "SymbiYosys",
            "certificate_hash": "",
        }
        values.update(kwargs)
        with pytest.raises(ValueError, match=match):
            FormalProofCertificate(**_unsafe(values))

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"prop_id": ""}, "prop_id"),
            ({"module": ""}, "module"),
            ({"description": ""}, "description"),
            ({"property_type": "prove"}, "property_type"),
            ({"status": "ok"}, "status"),
            ({"engine": ""}, "engine"),
            ({"depth": -1}, "depth"),
            ({"depth": True}, "depth"),
            ({"sby_file": None}, "sby_file"),
        ],
    )
    def test_formal_property_rejects_invalid_contracts(self, kwargs: Any, match: Any) -> None:
        values = {
            "prop_id": "P1",
            "module": "sc_lif_neuron",
            "description": "desc",
            "property_type": "assert",
            "status": "proven",
            "engine": "SymbiYosys",
            "depth": 20,
            "sby_file": "sc_lif_neuron.sby",
        }
        values.update(kwargs)
        with pytest.raises(ValueError, match=match):
            FormalProperty(**_unsafe(values))


class TestProofTestCoverage:
    def test_full_coverage(self) -> None:
        props = [
            FormalProperty("P1", "m", "d", "assert", "proven"),
            FormalProperty("P2", "m", "d", "assert", "proven"),
        ]
        assert ProofTestCoverage.coverage_from_proofs(props) == 1.0

    def test_partial_coverage(self) -> None:
        props = [
            FormalProperty("P1", "m", "d", "assert", "proven"),
            FormalProperty("P2", "m", "d", "assert", "failed"),
        ]
        assert abs(ProofTestCoverage.coverage_from_proofs(props) - 0.5) < 0.01

    def test_uncovered_modules(self) -> None:
        props = [FormalProperty("P1", "neuron", "d", "assert", "proven")]
        uncovered = ProofTestCoverage.uncovered_modules(props, ["neuron", "encoder"])
        assert uncovered == ["encoder"]

    def test_uncovered_modules_deduplicates_preserving_order(self) -> None:
        props = [FormalProperty("P1", "neuron", "d", "assert", "proven")]
        uncovered = ProofTestCoverage.uncovered_modules(props, ["encoder", "encoder", "decoder"])
        assert uncovered == ["encoder", "decoder"]

    def test_dc_to_sil(self) -> None:
        assert ProofTestCoverage.dc_to_sil(0.99).value >= 3
        assert ProofTestCoverage.dc_to_sil(0.97) == SILLevel.SIL_3
        assert ProofTestCoverage.dc_to_sil(0.5) == SILLevel.SIL_1

    @pytest.mark.parametrize("dc", [-0.1, 1.1, float("nan"), float("inf"), True, "0.9"])
    def test_dc_to_sil_rejects_invalid_contracts(self, dc: Any) -> None:
        with pytest.raises(ValueError, match="dc"):
            ProofTestCoverage.dc_to_sil(_unsafe(dc))

    @pytest.mark.parametrize(
        ("props", "modules", "match"),
        [
            ("invalid", ["neuron"], "properties"),
            ([FormalProperty("P1", "n", "d", "assert", "proven"), "bad"], ["neuron"], "properties"),
            ([FormalProperty("P1", "n", "d", "assert", "proven")], "invalid", "all_modules"),
            ([FormalProperty("P1", "n", "d", "assert", "proven")], ["", "neuron"], "all_modules"),
            ([FormalProperty("P1", "n", "d", "assert", "proven")], [" neuron"], "whitespace"),
        ],
    )
    def test_uncovered_modules_rejects_invalid_contracts(
        self, props: Any, modules: Any, match: Any
    ) -> None:
        with pytest.raises(ValueError, match=match):
            ProofTestCoverage.uncovered_modules(_unsafe(props), _unsafe(modules))

    def test_uncovered_modules_rejects_corrupted_property_module(self) -> None:
        prop = FormalProperty("P1", "m", "d", "assert", "proven")
        prop.module = _unsafe("")
        with pytest.raises(ValueError, match="modules"):
            ProofTestCoverage.uncovered_modules([prop], ["m"])

    def test_uncovered_modules_rejects_corrupted_property_id(self) -> None:
        prop = FormalProperty("P1", "m", "d", "assert", "proven")
        prop.prop_id = _unsafe("")
        with pytest.raises(ValueError, match="prop_id"):
            ProofTestCoverage.uncovered_modules([prop], ["m"])

    @pytest.mark.parametrize(
        "props", ["invalid", [FormalProperty("P1", "n", "d", "assert", "proven"), "bad"]]
    )
    def test_coverage_from_proofs_rejects_invalid_contracts(self, props: Any) -> None:
        with pytest.raises(ValueError, match="properties"):
            ProofTestCoverage.coverage_from_proofs(_unsafe(props))

    def test_coverage_from_proofs_rejects_corrupted_property_status(self) -> None:
        prop = FormalProperty("P1", "m", "d", "assert", "proven")
        prop.status = _unsafe("bad")
        with pytest.raises(ValueError, match="statuses"):
            ProofTestCoverage.coverage_from_proofs([prop])

    def test_coverage_from_proofs_rejects_corrupted_property_type(self) -> None:
        prop = FormalProperty("P1", "m", "d", "assert", "proven")
        prop.property_type = _unsafe("bad")
        with pytest.raises(ValueError, match="property_type"):
            ProofTestCoverage.coverage_from_proofs([prop])

    def test_coverage_from_proofs_rejects_corrupted_property_id(self) -> None:
        prop = FormalProperty("P1", "m", "d", "assert", "proven")
        prop.prop_id = _unsafe("")
        with pytest.raises(ValueError, match="prop_id"):
            ProofTestCoverage.coverage_from_proofs([prop])


class TestFormalGapDetector:
    def test_fully_covered(self) -> None:
        props = [
            FormalProperty("P1", "neuron", "d", "assert", "proven"),
            FormalProperty("P2", "neuron", "d", "cover", "proven"),
        ]
        assert FormalPropertyGapDetector.is_fully_covered(props, ["neuron"])

    def test_missing_module(self) -> None:
        props = [FormalProperty("P1", "neuron", "d", "assert", "proven")]
        gaps = FormalPropertyGapDetector.detect(props, ["neuron", "encoder"])
        assert len(gaps) >= 1
        assert any(g.module == "encoder" for g in gaps)

    def test_detect_deduplicates_required_modules(self) -> None:
        props = [FormalProperty("P1", "neuron", "d", "assert", "proven")]
        gaps = FormalPropertyGapDetector.detect(props, ["encoder", "encoder"])
        assert [g.module for g in gaps] == ["encoder"]

    def test_failed_property(self) -> None:
        props = [
            FormalProperty("P1", "neuron", "d", "assert", "failed"),
            FormalProperty("P2", "neuron", "d", "cover", "proven"),
        ]
        gaps = FormalPropertyGapDetector.detect(props, ["neuron"])
        assert len(gaps) == 1
        assert gaps[0].proven_properties == 1

    def test_gap_coverage(self) -> None:
        gap = PropertyGap("m", 4, 2, [])
        assert gap.coverage == 0.5

    @pytest.mark.parametrize(
        ("properties", "required_modules", "match"),
        [
            ("bad", ["neuron"], "properties"),
            ([FormalProperty("P1", "n", "d", "assert", "proven"), "bad"], ["neuron"], "properties"),
            ([FormalProperty("P1", "n", "d", "assert", "proven")], "bad", "required_modules"),
            ([FormalProperty("P1", "n", "d", "assert", "proven")], [""], "required_modules"),
            ([FormalProperty("P1", "n", "d", "assert", "proven")], [" neuron"], "whitespace"),
        ],
    )
    def test_detect_rejects_invalid_contracts(
        self, properties: Any, required_modules: Any, match: Any
    ) -> None:
        with pytest.raises(ValueError, match=match):
            FormalPropertyGapDetector.detect(_unsafe(properties), _unsafe(required_modules))

    def test_detect_rejects_corrupted_property_type_state(self) -> None:
        prop = FormalProperty("P1", "neuron", "d", "assert", "proven")
        prop.property_type = _unsafe("bad")
        with pytest.raises(ValueError, match="property_type"):
            FormalPropertyGapDetector.detect([prop], ["neuron"])

    def test_detect_rejects_corrupted_property_status_state(self) -> None:
        prop = FormalProperty("P1", "neuron", "d", "assert", "proven")
        prop.status = _unsafe("bad")
        with pytest.raises(ValueError, match="statuses"):
            FormalPropertyGapDetector.detect([prop], ["neuron"])

    def test_detect_rejects_corrupted_property_module_state(self) -> None:
        prop = FormalProperty("P1", "neuron", "d", "assert", "proven")
        prop.module = _unsafe("")
        with pytest.raises(ValueError, match="modules"):
            FormalPropertyGapDetector.detect([prop], ["neuron"])

    def test_detect_rejects_corrupted_property_id_state(self) -> None:
        prop = FormalProperty("P1", "neuron", "d", "assert", "proven")
        prop.prop_id = _unsafe("")
        with pytest.raises(ValueError, match="prop_id"):
            FormalPropertyGapDetector.detect([prop], ["neuron"])

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"module": ""}, "module"),
            ({"total_properties": -1}, "total_properties"),
            ({"total_properties": True}, "total_properties"),
            ({"proven_properties": -1}, "proven_properties"),
            ({"proven_properties": 3, "total_properties": 2}, "proven_properties cannot exceed"),
            ({"missing_types": ["", "cover"]}, "missing_types"),
        ],
    )
    def test_property_gap_rejects_invalid_contracts(self, kwargs: Any, match: Any) -> None:
        values = {
            "module": "neuron",
            "total_properties": 2,
            "proven_properties": 1,
            "missing_types": ["assert"],
        }
        values.update(kwargs)
        with pytest.raises(ValueError, match=match):
            PropertyGap(**_unsafe(values))


class TestBoundaryContracts:
    def test_formal_certificate_add_property_and_report_validation(self) -> None:
        cert = FormalProofCertificate()
        prop = FormalProperty("P1", "neuron", "desc", "assert", "proven")
        cert.add_property(prop)
        assert cert.total_count == 1
        prop.property_type = _unsafe("invalid")
        with pytest.raises(ValueError, match="property_type"):
            cert.generate_report()

    def test_proof_test_assessment_boundaries(self) -> None:
        assert (
            ProofTestCoverage.coverage_from_proofs(
                [FormalProperty("P1", "m", "desc", "cover", "proven")]
            )
            == 0.0
        )
        assert ProofTestCoverage.dc_to_sil(0.9) == SILLevel.SIL_2
        assert ProofTestCoverage.dc_to_sil(0.6) == SILLevel.SIL_1

    def test_property_gap_rejects_boolean_proven_count(self) -> None:
        with pytest.raises(ValueError, match="proven_properties"):
            PropertyGap("module", 2, True, ["assert"])


class TestFormalDigestBoundary:
    @pytest.mark.parametrize(
        ("field", "value", "match"),
        [
            ("prop_id", "", "prop_id"),
            ("description", "", "descriptions"),
            ("property_type", "invalid", "property_type"),
            ("status", "invalid", "statuses"),
            ("engine", "", "engines"),
            ("depth", True, "depths"),
            ("sby_file", None, "sby_file"),
        ],
    )
    def test_content_digest_rejects_corrupted_material_fields(
        self,
        field: str,
        value: object,
        match: str,
    ) -> None:
        prop = FormalProperty("P1", "neuron", "description", "assert", "proven")
        setattr(prop, field, value)
        with pytest.raises(ValueError, match=match):
            FormalProofCertificate([prop]).content_sha256()

    def test_hash_and_report_reject_empty_explicit_timestamp(self) -> None:
        cert = FormalProofCertificate(
            [FormalProperty("P1", "neuron", "description", "assert", "proven")]
        )
        with pytest.raises(ValueError, match="generated_at"):
            cert.compute_hash(generated_at="")
        cert.compute_hash(generated_at="2026-07-12T18:30:00+00:00")
        assert "Formal Proof Certificate" in cert.generate_report()
        with pytest.raises(ValueError, match="generated_at"):
            cert.generate_report(generated_at="")

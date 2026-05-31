# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Safety Certification Generator Tests

import pytest

from sc_neurocore.safety_cert.safety_cert import (
    ASILLevel,
    CCFAnalysis,
    CCFDefence,
    CertificationGenerator,
    CertificationPackage,
    ChangeImpactTracker,
    ChangeRecord,
    ChecklistItem,
    ComplianceChecklist,
    CrossStandardMapper,
    EvidenceBag,
    EvidenceItem,
    FailureCategory,
    FailureMode,
    FMEDA,
    FormalPropertyGapDetector,
    FormalProofCertificate,
    FormalProperty,
    HFTAssessment,
    HFTLevel,
    IEC62304Assessment,
    ProofTestCoverage,
    PropertyGap,
    ReliabilityMetrics,
    Requirement,
    SafetyManualGenerator,
    SafetyStandard,
    SILLevel,
    SIL_TO_ASIL,
    SWClass,
    TraceabilityMatrix,
    WCETPath,
    WCETAnalyzer,
    CROSS_MAP,
)


# ── TraceabilityMatrix Tests ─────────────────────────────────────────


class TestTraceabilityMatrix:
    def test_add_requirement(self):
        tm = TraceabilityMatrix()
        req = Requirement("REQ_001", "Test", SafetyStandard.IEC_61508)
        tm.add_requirement(req)
        assert "REQ_001" in tm.requirements

    def test_add_requirement_rejects_duplicate_req_id(self):
        tm = TraceabilityMatrix()
        req = Requirement("REQ_001", "Test", SafetyStandard.IEC_61508)
        tm.add_requirement(req)
        with pytest.raises(ValueError, match="already exists"):
            tm.add_requirement(req)

    def test_link_implementation(self):
        tm = TraceabilityMatrix()
        tm.add_requirement(Requirement("REQ_001", "Test", SafetyStandard.IEC_61508))
        assert tm.link_implementation("REQ_001", "hdl/test.v") is True
        assert tm.requirements["REQ_001"].status == "implemented"

    def test_link_implementation_normalises_whitespace_inputs(self):
        tm = TraceabilityMatrix()
        tm.add_requirement(Requirement("REQ_001", "Test", SafetyStandard.IEC_61508))
        assert tm.link_implementation(" REQ_001 ", " hdl/test.v ") is True
        assert tm.requirements["REQ_001"].implementation_refs == ["hdl/test.v"]

    def test_link_implementation_is_idempotent_for_same_reference(self):
        tm = TraceabilityMatrix()
        tm.add_requirement(Requirement("REQ_001", "Test", SafetyStandard.IEC_61508))
        tm.link_implementation("REQ_001", "hdl/test.v")
        tm.link_implementation("REQ_001", "hdl/test.v")
        assert tm.requirements["REQ_001"].implementation_refs == ["hdl/test.v"]

    def test_link_implementation_rejects_corrupted_requirement_entry(self):
        tm = TraceabilityMatrix()
        tm.requirements["REQ_001"] = "bad"  # type: ignore[assignment]
        with pytest.raises(ValueError, match="Requirement"):
            tm.link_implementation("REQ_001", "hdl/test.v")

    def test_link_verification(self):
        tm = TraceabilityMatrix()
        req = Requirement("REQ_001", "Test", SafetyStandard.IEC_61508)
        req.implementation_refs = ["hdl/test.v"]
        tm.add_requirement(req)
        tm.link_verification("REQ_001", "formal/test.sby")
        assert tm.requirements["REQ_001"].status == "verified"

    def test_link_verification_normalises_whitespace_inputs(self):
        tm = TraceabilityMatrix()
        req = Requirement("REQ_001", "Test", SafetyStandard.IEC_61508)
        req.implementation_refs = ["hdl/test.v"]
        tm.add_requirement(req)
        tm.link_verification(" REQ_001 ", " formal/test.sby ")
        assert tm.requirements["REQ_001"].verification_refs == ["formal/test.sby"]

    def test_link_verification_is_idempotent_for_same_reference(self):
        tm = TraceabilityMatrix()
        req = Requirement("REQ_001", "Test", SafetyStandard.IEC_61508)
        req.implementation_refs = ["hdl/test.v"]
        tm.add_requirement(req)
        tm.link_verification("REQ_001", "formal/test.sby")
        tm.link_verification("REQ_001", "formal/test.sby")
        assert tm.requirements["REQ_001"].verification_refs == ["formal/test.sby"]

    def test_link_verification_rejects_corrupted_requirement_entry(self):
        tm = TraceabilityMatrix()
        tm.requirements["REQ_001"] = "bad"  # type: ignore[assignment]
        with pytest.raises(ValueError, match="Requirement"):
            tm.link_verification("REQ_001", "formal/test.sby")

    def test_coverage(self):
        tm = TraceabilityMatrix()
        for i in range(4):
            req = Requirement(f"REQ_{i}", "Test", SafetyStandard.IEC_61508)
            if i < 2:
                req.implementation_refs = ["impl"]
                req.verification_refs = ["verif"]
                req.status = "verified"
            tm.add_requirement(req)
        assert abs(tm.coverage - 0.5) < 0.01

    def test_coverage_rejects_corrupted_internal_state(self):
        tm = TraceabilityMatrix()
        tm.requirements["R1"] = "bad"  # type: ignore[assignment]
        with pytest.raises(ValueError, match="Requirement"):
            _ = tm.coverage

    def test_coverage_rejects_corrupted_requirement_status(self):
        tm = TraceabilityMatrix()
        req = Requirement("R1", "Test", SafetyStandard.IEC_61508)
        req.status = "bad"  # type: ignore[assignment]
        tm.add_requirement(req)
        with pytest.raises(ValueError, match="statuses"):
            _ = tm.coverage

    def test_open_count(self):
        tm = TraceabilityMatrix()
        tm.add_requirement(Requirement("R1", "Test", SafetyStandard.IEC_61508))
        assert tm.open_count == 1

    def test_open_count_rejects_corrupted_internal_state(self):
        tm = TraceabilityMatrix()
        tm.requirements["R1"] = "bad"  # type: ignore[assignment]
        with pytest.raises(ValueError, match="Requirement"):
            _ = tm.open_count

    def test_open_count_rejects_corrupted_requirement_status(self):
        tm = TraceabilityMatrix()
        req = Requirement("R1", "Test", SafetyStandard.IEC_61508)
        req.status = "bad"  # type: ignore[assignment]
        tm.add_requirement(req)
        with pytest.raises(ValueError, match="statuses"):
            _ = tm.open_count

    def test_link_nonexistent(self):
        tm = TraceabilityMatrix()
        assert tm.link_implementation("NOPE", "x.v") is False

    def test_generate_report(self):
        tm = TraceabilityMatrix()
        tm.add_requirement(Requirement("R1", "Test", SafetyStandard.IEC_61508))
        report = tm.generate_report()
        assert "Traceability Matrix" in report
        assert "R1" in report

    def test_generate_report_orders_rows_by_requirement_id(self):
        tm = TraceabilityMatrix()
        tm.add_requirement(Requirement("R2", "Test", SafetyStandard.IEC_61508))
        tm.add_requirement(Requirement("R1", "Test", SafetyStandard.IEC_61508))
        lines = tm.generate_report().splitlines()
        req_rows = [
            line for line in lines if line.startswith("| R") and not line.startswith("| Req ID")
        ]
        assert req_rows == sorted(req_rows)

    def test_generate_report_rejects_corrupted_internal_state(self):
        tm = TraceabilityMatrix()
        tm.requirements["R1"] = "bad"  # type: ignore[assignment]
        with pytest.raises(ValueError, match="Requirement"):
            tm.generate_report()

    def test_generate_report_rejects_corrupted_requirement_status(self):
        tm = TraceabilityMatrix()
        req = Requirement("R1", "Test", SafetyStandard.IEC_61508)
        req.status = "bad"  # type: ignore[assignment]
        tm.add_requirement(req)
        with pytest.raises(ValueError, match="statuses"):
            tm.generate_report()

    @pytest.mark.parametrize(
        "property_name",
        ["coverage", "open_count"],
    )
    def test_traceability_properties_reject_requirement_key_mismatch(self, property_name):
        tm = TraceabilityMatrix()
        tm.requirements["R1"] = Requirement("R2", "Test", SafetyStandard.IEC_61508)
        with pytest.raises(ValueError, match="key mismatch"):
            _ = getattr(tm, property_name)

    def test_generate_report_rejects_requirement_key_mismatch(self):
        tm = TraceabilityMatrix()
        tm.requirements["R1"] = Requirement("R2", "Test", SafetyStandard.IEC_61508)
        with pytest.raises(ValueError, match="key mismatch"):
            tm.generate_report()

    @pytest.mark.parametrize(
        ("field_name", "bad_value", "match"),
        [
            ("standard", "IEC 61508", "SafetyStandard"),
            ("sil_level", 2, "SILLevel"),
        ],
    )
    def test_generate_report_rejects_invalid_requirement_types(self, field_name, bad_value, match):
        tm = TraceabilityMatrix()
        req = Requirement("R1", "Test", SafetyStandard.IEC_61508)
        setattr(req, field_name, bad_value)
        tm.add_requirement(req)
        with pytest.raises(ValueError, match=match):
            tm.generate_report()

    def test_add_requirement_rejects_invalid_contract(self):
        tm = TraceabilityMatrix()
        with pytest.raises(ValueError, match="req"):
            tm.add_requirement("bad")  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        ("req_id", "impl_ref", "match"),
        [
            ("", "hdl/a.v", "req_id"),
            ("REQ_1", "", "impl_ref"),
        ],
    )
    def test_link_implementation_rejects_invalid_contracts(self, req_id, impl_ref, match):
        tm = TraceabilityMatrix()
        with pytest.raises(ValueError, match=match):
            tm.link_implementation(req_id, impl_ref)

    @pytest.mark.parametrize(
        ("req_id", "verif_ref", "match"),
        [
            ("", "formal/a.sby", "req_id"),
            ("REQ_1", "", "verif_ref"),
        ],
    )
    def test_link_verification_rejects_invalid_contracts(self, req_id, verif_ref, match):
        tm = TraceabilityMatrix()
        with pytest.raises(ValueError, match=match):
            tm.link_verification(req_id, verif_ref)

    def test_update_status_rejects_invalid_requirement_object(self):
        tm = TraceabilityMatrix()
        with pytest.raises(ValueError, match="req"):
            tm._update_status("bad")  # type: ignore[arg-type]

    def test_update_status_rejects_corrupted_reference_entries(self):
        tm = TraceabilityMatrix()
        req = Requirement("R1", "d", SafetyStandard.IEC_61508)
        req.implementation_refs = [""]  # type: ignore[assignment]
        with pytest.raises(ValueError, match="implementation_refs"):
            tm._update_status(req)

    def test_update_status_downgrades_to_open_when_implementation_removed(self):
        tm = TraceabilityMatrix()
        req = Requirement("R1", "d", SafetyStandard.IEC_61508)
        req.status = "verified"
        req.implementation_refs = []
        req.verification_refs = ["formal/test.sby"]
        tm._update_status(req)
        assert req.status == "open"

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"req_id": ""}, "req_id"),
            ({"description": ""}, "description"),
            ({"standard": "IEC 61508"}, "standard"),
            ({"sil_level": 2}, "sil_level"),
            ({"status": ""}, "status"),
            ({"status": "done"}, "status"),
            ({"implementation_refs": ["", "hdl/top.sv"]}, "implementation_refs"),
            ({"verification_refs": ["", "formal/top.sby"]}, "verification_refs"),
        ],
    )
    def test_requirement_rejects_invalid_contracts(self, kwargs, match):
        values = {
            "req_id": "REQ_100",
            "description": "desc",
            "standard": SafetyStandard.IEC_61508,
            "sil_level": SILLevel.SIL_2,
            "implementation_refs": ["hdl/top.sv"],
            "verification_refs": ["formal/top.sby"],
            "status": "open",
        }
        values.update(kwargs)
        with pytest.raises(ValueError, match=match):
            Requirement(**values)


# ── FMEDA Tests ──────────────────────────────────────────────────────


class TestFMEDA:
    def test_add_failure_mode(self):
        fmeda = FMEDA()
        fm = FailureMode("FM1", "neuron", "stuck", FailureCategory.SAFE, 10.0)
        fmeda.add_failure_mode(fm)
        assert len(fmeda.failure_modes) == 1

    def test_add_failure_mode_rejects_duplicate_fm_id(self):
        fmeda = FMEDA()
        fmeda.add_failure_mode(FailureMode("FM1", "neuron", "stuck", FailureCategory.SAFE, 10.0))
        with pytest.raises(ValueError, match="already exists"):
            fmeda.add_failure_mode(
                FailureMode("FM1", "neuron", "other", FailureCategory.SAFE, 12.0)
            )

    def test_add_failure_mode_rejects_invalid_contract(self):
        fmeda = FMEDA()
        with pytest.raises(ValueError, match="fm"):
            fmeda.add_failure_mode("bad")  # type: ignore[arg-type]

    def test_add_sc_standard_modes(self):
        fmeda = FMEDA()
        fmeda.add_sc_standard_modes("sc_lif_neuron")
        assert len(fmeda.failure_modes) == 5

    def test_add_sc_standard_modes_normalises_component_whitespace(self):
        fmeda = FMEDA()
        fmeda.add_sc_standard_modes(" sc_lif_neuron ")
        assert all(fm.component == "sc_lif_neuron" for fm in fmeda.failure_modes)

    def test_add_sc_standard_modes_rejects_duplicate_component_seed(self):
        fmeda = FMEDA()
        fmeda.add_sc_standard_modes("neuron")
        with pytest.raises(ValueError, match="already exists"):
            fmeda.add_sc_standard_modes("neuron")

    def test_add_sc_standard_modes_rejects_invalid_component(self):
        fmeda = FMEDA()
        with pytest.raises(ValueError, match="component"):
            fmeda.add_sc_standard_modes("")

    def test_total_failure_rate(self):
        fmeda = FMEDA()
        fmeda.add_sc_standard_modes("neuron")
        assert fmeda.total_failure_rate > 0

    def test_total_failure_rate_rejects_corrupted_internal_state(self):
        fmeda = FMEDA()
        fmeda.failure_modes.append("bad")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="FailureMode"):
            _ = fmeda.total_failure_rate

    def test_total_failure_rate_rejects_corrupted_total_state(self):
        fmeda = FMEDA()
        fm = FailureMode("FM1", "n", "d", FailureCategory.SAFE, 1.0)
        fm.failure_rate_fit = float("nan")  # type: ignore[assignment]
        fmeda.add_failure_mode(fm)
        with pytest.raises(ValueError, match="failure_rate_fit"):
            _ = fmeda.total_failure_rate

    def test_safe_failure_fraction(self):
        fmeda = FMEDA()
        fmeda.add_sc_standard_modes("neuron")
        sff = fmeda.safe_failure_fraction
        assert 0.0 < sff <= 1.0

    def test_safe_failure_fraction_rejects_corrupted_internal_state(self):
        fmeda = FMEDA()
        fmeda.failure_modes.append("bad")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="FailureMode"):
            _ = fmeda.safe_failure_fraction

    def test_diagnostic_coverage(self):
        fmeda = FMEDA()
        fmeda.add_sc_standard_modes("neuron")
        dc = fmeda.diagnostic_coverage
        assert 0.0 < dc <= 1.0

    def test_diagnostic_coverage_rejects_corrupted_internal_state(self):
        fmeda = FMEDA()
        fmeda.failure_modes.append("bad")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="FailureMode"):
            _ = fmeda.diagnostic_coverage

    def test_diagnostic_coverage_rejects_corrupted_aggregate_state(self):
        fmeda = FMEDA()
        fm = FailureMode("FM1", "n", "d", FailureCategory.DANGEROUS_DETECTED, 1.0, 0.5)
        fm.failure_rate_fit = float("nan")  # type: ignore[assignment]
        fmeda.add_failure_mode(fm)
        with pytest.raises(ValueError, match="denominator|weighted sum"):
            _ = fmeda.diagnostic_coverage

    def test_diagnostic_coverage_rejects_corrupted_entry_coverage_value(self):
        fmeda = FMEDA()
        fm = FailureMode("FM1", "n", "d", FailureCategory.DANGEROUS_DETECTED, 1.0, 0.5)
        fm.diagnostic_coverage = 1.5  # type: ignore[assignment]
        fmeda.add_failure_mode(fm)
        with pytest.raises(ValueError, match="entries must be in"):
            _ = fmeda.diagnostic_coverage

    def test_sff_by_component_rejects_corrupted_internal_state(self):
        fmeda = FMEDA()
        fmeda.failure_modes.append("bad")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="FailureMode"):
            fmeda.sff_by_component()

    def test_sff_by_component_rejects_corrupted_component_totals(self):
        fmeda = FMEDA()
        fm = FailureMode("FM1", "n", "d", FailureCategory.DANGEROUS_DETECTED, 1.0, 0.5)
        fm.diagnostic_coverage = float("nan")  # type: ignore[assignment]
        fmeda.add_failure_mode(fm)
        with pytest.raises(ValueError, match="totals"):
            fmeda.sff_by_component()

    def test_sff_by_component_rejects_empty_component_name(self):
        fmeda = FMEDA()
        fm = FailureMode("FM1", "n", "d", FailureCategory.SAFE, 1.0)
        fm.component = ""  # type: ignore[assignment]
        fmeda.add_failure_mode(fm)
        with pytest.raises(ValueError, match="component names"):
            fmeda.sff_by_component()

    def test_max_sil(self):
        fmeda = FMEDA()
        fmeda.add_sc_standard_modes("neuron")
        sil = fmeda.max_achievable_sil()
        assert sil.value >= 1

    def test_max_sil_rejects_corrupted_coverage_state(self):
        fmeda = FMEDA()
        fm = FailureMode("FM1", "n", "d", FailureCategory.DANGEROUS_DETECTED, 1.0, 0.5)
        fm.diagnostic_coverage = float("nan")  # type: ignore[assignment]
        fmeda.add_failure_mode(fm)
        with pytest.raises(ValueError, match="coverage|safe_failure_fraction"):
            fmeda.max_achievable_sil()

    def test_generate_report(self):
        fmeda = FMEDA()
        fmeda.add_sc_standard_modes("neuron")
        report = fmeda.generate_report()
        assert "FMEDA" in report
        assert "FIT" in report

    def test_generate_report_orders_rows_by_failure_mode_id(self):
        fmeda = FMEDA()
        fmeda.add_failure_mode(FailureMode("FM2", "n", "d", FailureCategory.SAFE, 1.0))
        fmeda.add_failure_mode(FailureMode("FM1", "n", "d", FailureCategory.SAFE, 1.0))
        lines = fmeda.generate_report().splitlines()
        rows = [
            line for line in lines if line.startswith("| FM") and not line.startswith("| FM ID")
        ]
        assert rows == sorted(rows)

    def test_generate_report_rejects_corrupted_internal_state(self):
        fmeda = FMEDA()
        fmeda.failure_modes.append("bad")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="FailureMode"):
            fmeda.generate_report()

    def test_safe_failure_fraction_value(self):
        fmeda = FMEDA()
        fm = FailureMode("FM1", "x", "safe", FailureCategory.SAFE, 100.0)
        fmeda.add_failure_mode(fm)
        assert fmeda.safe_failure_fraction == 1.0

    def test_sff_all_dangerous(self):
        fmeda = FMEDA()
        fm = FailureMode("FM1", "x", "bad", FailureCategory.DANGEROUS_UNDETECTED, 100.0)
        fmeda.add_failure_mode(fm)
        assert fmeda.safe_failure_fraction == 0.0

    def test_residual_risk_rejects_corrupted_internal_state(self):
        fmeda = FMEDA()
        fmeda.failure_modes.append("bad")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="FailureMode"):
            _ = fmeda.residual_risk_fit

    def test_residual_risk_rejects_corrupted_aggregate_state(self):
        fmeda = FMEDA()
        fm = FailureMode("FM1", "n", "d", FailureCategory.DANGEROUS_DETECTED, 1.0, 0.5)
        fm.diagnostic_coverage = float("nan")  # type: ignore[assignment]
        fmeda.add_failure_mode(fm)
        with pytest.raises(ValueError, match="residual_risk_fit"):
            _ = fmeda.residual_risk_fit

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"fm_id": ""}, "fm_id"),
            ({"component": ""}, "component"),
            ({"description": ""}, "description"),
            ({"category": "safe"}, "category"),
            ({"failure_rate_fit": -1.0}, "failure_rate_fit"),
            ({"failure_rate_fit": float("nan")}, "failure_rate_fit"),
            ({"failure_rate_fit": True}, "failure_rate_fit"),
            ({"diagnostic_coverage": -0.1}, "diagnostic_coverage"),
            ({"diagnostic_coverage": 1.1}, "diagnostic_coverage"),
            ({"diagnostic_coverage": float("inf")}, "diagnostic_coverage"),
            ({"diagnostic_coverage": False}, "diagnostic_coverage"),
            ({"mitigation": None}, "mitigation"),
        ],
    )
    def test_failure_mode_rejects_invalid_contracts(self, kwargs, match):
        values = {
            "fm_id": "FM1",
            "component": "neuron",
            "description": "desc",
            "category": FailureCategory.SAFE,
            "failure_rate_fit": 1.0,
            "diagnostic_coverage": 0.5,
            "mitigation": "mitigate",
        }
        values.update(kwargs)
        with pytest.raises(ValueError, match=match):
            FailureMode(**values)


# ── FormalProofCertificate Tests ─────────────────────────────────────


class TestFormalProofCertificate:
    def _props(self):
        return [
            FormalProperty("P1", "sc_lif_neuron", "No overflow", "assert", "proven"),
            FormalProperty("P2", "sc_lif_neuron", "Reset works", "assert", "proven"),
            FormalProperty("P3", "sc_encoder", "Cover fire", "cover", "proven"),
            FormalProperty("P4", "sc_dense", "Weight range", "assert", "failed"),
        ]

    def test_proven_count(self):
        cert = FormalProofCertificate(properties=self._props())
        assert cert.proven_count == 3

    def test_proven_count_rejects_corrupted_internal_state(self):
        cert = FormalProofCertificate()
        cert.properties.append("bad")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="FormalProperty"):
            _ = cert.proven_count

    def test_proven_count_rejects_corrupted_property_status(self):
        cert = FormalProofCertificate()
        prop = FormalProperty("P1", "m", "d", "assert", "proven")
        prop.status = "bad"  # type: ignore[assignment]
        cert.properties.append(prop)
        with pytest.raises(ValueError, match="statuses"):
            _ = cert.proven_count

    def test_pass_rate(self):
        cert = FormalProofCertificate(properties=self._props())
        assert abs(cert.pass_rate - 0.75) < 0.01

    def test_total_count_rejects_corrupted_internal_state(self):
        cert = FormalProofCertificate()
        cert.properties.append("bad")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="FormalProperty"):
            _ = cert.total_count

    def test_total_count_rejects_corrupted_property_id(self):
        cert = FormalProofCertificate()
        prop = FormalProperty("P1", "m", "d", "assert", "proven")
        prop.prop_id = ""  # type: ignore[assignment]
        cert.properties.append(prop)
        with pytest.raises(ValueError, match="prop_id"):
            _ = cert.total_count

    def test_compute_hash(self):
        cert = FormalProofCertificate(properties=self._props())
        h = cert.compute_hash()
        assert len(h) == 32

    def test_add_property_rejects_invalid_contract(self):
        cert = FormalProofCertificate()
        with pytest.raises(ValueError, match="prop"):
            cert.add_property("bad")  # type: ignore[arg-type]

    def test_hash_deterministic(self):
        cert = FormalProofCertificate(properties=self._props())
        assert cert.compute_hash() == cert.compute_hash()

    def test_compute_hash_rejects_duplicate_property_ids(self):
        cert = FormalProofCertificate(
            properties=[
                FormalProperty("P1", "m1", "d1", "assert", "proven"),
                FormalProperty("P1", "m2", "d2", "assert", "proven"),
            ]
        )
        with pytest.raises(ValueError, match="duplicate"):
            cert.compute_hash()

    def test_compute_hash_rejects_corrupted_internal_state(self):
        cert = FormalProofCertificate()
        cert.properties.append("bad")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="FormalProperty"):
            cert.compute_hash()

    def test_compute_hash_rejects_corrupted_property_module(self):
        cert = FormalProofCertificate()
        prop = FormalProperty("P1", "m", "d", "assert", "proven")
        prop.module = ""  # type: ignore[assignment]
        cert.properties.append(prop)
        with pytest.raises(ValueError, match="modules"):
            cert.compute_hash()

    def test_generate_report(self):
        cert = FormalProofCertificate(properties=self._props())
        report = cert.generate_report()
        assert "Formal Proof Certificate" in report
        assert "P1" in report

    def test_generate_report_rejects_corrupted_internal_state(self):
        cert = FormalProofCertificate()
        cert.properties.append("bad")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="FormalProperty"):
            cert.generate_report()

    def test_generate_report_rejects_corrupted_property_fields(self):
        cert = FormalProofCertificate()
        prop = FormalProperty("P1", "m", "d", "assert", "proven")
        prop.prop_id = ""  # type: ignore[assignment]
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
    def test_formal_proof_certificate_rejects_invalid_contracts(self, kwargs, match):
        values = {
            "properties": self._props(),
            "generation_timestamp": "",
            "tool_version": "SymbiYosys",
            "certificate_hash": "",
        }
        values.update(kwargs)
        with pytest.raises(ValueError, match=match):
            FormalProofCertificate(**values)

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
    def test_formal_property_rejects_invalid_contracts(self, kwargs, match):
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
            FormalProperty(**values)


# ── WCET Tests ───────────────────────────────────────────────────────


class TestWCETAnalyzer:
    def test_basic_analysis(self):
        path = WCETAnalyzer.analyze(256, 8, 16)
        assert path.total_cycles > 0
        assert len(path.stages) == 4

    def test_wcet_ns(self):
        path = WCETAnalyzer.analyze(256, 8, 16)
        ns = path.wcet_ns(100.0)
        assert ns > 0

    def test_total_cycles_rejects_corrupted_internal_state(self):
        path = WCETPath("p1", "path", ["A"], [1])
        path.cycles_per_stage = ["bad"]  # type: ignore[assignment]
        with pytest.raises(ValueError, match="cycles_per_stage"):
            _ = path.total_cycles

    def test_with_stp(self):
        path = WCETAnalyzer.analyze(256, 8, 16, has_stp=True)
        assert len(path.stages) == 5
        assert "STP_Update" in path.stages

    def test_scaling(self):
        small = WCETAnalyzer.analyze(128, 4, 8)
        large = WCETAnalyzer.analyze(1024, 64, 128)
        assert large.total_cycles > small.total_cycles

    def test_multistage(self):
        layers = [
            {"bitstream_length": 256, "num_inputs": 8, "num_neurons": 16},
            {"bitstream_length": 256, "num_inputs": 16, "num_neurons": 4},
        ]
        path = WCETAnalyzer.analyze_multistage(layers)
        assert len(path.stages) == 8
        assert path.total_cycles > 0

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"path_id": ""}, "path_id"),
            ({"description": ""}, "description"),
            ({"stages": [], "cycles_per_stage": []}, "stages must not be empty"),
            ({"stages": ["A", ""], "cycles_per_stage": [1, 2]}, "stages"),
            ({"stages": ["A"], "cycles_per_stage": [1, 2]}, "same length"),
            ({"cycles_per_stage": [1, -1]}, "cycles_per_stage"),
        ],
    )
    def test_wcet_path_rejects_invalid_contracts(self, kwargs, match):
        values = {
            "path_id": "p1",
            "description": "path",
            "stages": ["A", "B"],
            "cycles_per_stage": [1, 2],
        }
        values.update(kwargs)
        with pytest.raises(ValueError, match=match):
            WCETPath(**values)

    @pytest.mark.parametrize("clock_mhz", [0.0, -1.0, float("inf"), float("nan"), True])
    def test_wcet_ns_rejects_invalid_clock(self, clock_mhz):
        path = WCETPath("p1", "path", ["A"], [1])
        with pytest.raises(ValueError, match="clock_mhz"):
            path.wcet_ns(clock_mhz)

    @pytest.mark.parametrize(
        ("args", "match"),
        [
            ((0, 8, 16, False), "bitstream_length"),
            ((256, 0, 16, False), "num_inputs"),
            ((256, 8, 0, False), "num_neurons"),
            ((256, 8, 16, "yes"), "has_stp"),
        ],
    )
    def test_analyze_rejects_invalid_contracts(self, args, match):
        with pytest.raises(ValueError, match=match):
            WCETAnalyzer.analyze(*args)

    @pytest.mark.parametrize(
        ("layers", "match"),
        [
            ([], "non-empty list"),
            ([None], "dictionary"),
            ([{"bitstream_length": 0}], "bitstream_length"),
            ([{"num_inputs": 0}], "num_inputs"),
            ([{"num_neurons": 0}], "num_neurons"),
        ],
    )
    def test_analyze_multistage_rejects_invalid_contracts(self, layers, match):
        with pytest.raises(ValueError, match=match):
            WCETAnalyzer.analyze_multistage(layers)


# ── ComplianceChecklist Tests ────────────────────────────────────────


class TestComplianceChecklist:
    def test_iec_61508(self):
        items = ComplianceChecklist.generate(SafetyStandard.IEC_61508)
        assert len(items) == 7
        assert all(isinstance(i, ChecklistItem) for i in items)

    def test_iso_26262(self):
        items = ComplianceChecklist.generate(SafetyStandard.ISO_26262)
        assert len(items) == 7

    def test_fda_class_iii(self):
        items = ComplianceChecklist.generate(SafetyStandard.FDA_CLASS_III)
        assert len(items) == 7

    def test_do_254(self):
        items = ComplianceChecklist.generate(SafetyStandard.DO_254)
        assert len(items) == 6

    def test_en_50129(self):
        items = ComplianceChecklist.generate(SafetyStandard.EN_50129)
        assert len(items) == 6

    def test_items_have_evidence(self):
        items = ComplianceChecklist.generate(SafetyStandard.IEC_61508)
        assert all(i.evidence for i in items)

    def test_generate_rejects_invalid_standard(self):
        with pytest.raises(ValueError, match="standard"):
            ComplianceChecklist.generate("IEC 61508")  # type: ignore[arg-type]

    def test_generate_rejects_duplicate_clause_definitions(self):
        original = ComplianceChecklist.IEC_61508_CLAUSES
        try:
            ComplianceChecklist.IEC_61508_CLAUSES = [
                ("7.4.2", "A", "formal/"),
                ("7.4.2", "B", "formal/"),
            ]
            with pytest.raises(ValueError, match="duplicates"):
                ComplianceChecklist.generate(SafetyStandard.IEC_61508)
        finally:
            ComplianceChecklist.IEC_61508_CLAUSES = original

    def test_generate_rejects_corrupted_clause_definition_shape(self):
        original = ComplianceChecklist.IEC_61508_CLAUSES
        try:
            ComplianceChecklist.IEC_61508_CLAUSES = [
                ("7.4.2", "A", "formal/"),
                ("7.4.3", "B", ""),
            ]
            with pytest.raises(ValueError, match="clause definitions"):
                ComplianceChecklist.generate(SafetyStandard.IEC_61508)
        finally:
            ComplianceChecklist.IEC_61508_CLAUSES = original

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"item_id": ""}, "item_id"),
            ({"clause": ""}, "clause"),
            ({"description": ""}, "description"),
            ({"evidence": None}, "evidence"),
            ({"status": "ok"}, "status"),
        ],
    )
    def test_checklist_item_rejects_invalid_contracts(self, kwargs, match):
        values = {
            "item_id": "IEC 61508_7.4.2",
            "clause": "7.4.2",
            "description": "Formal verification of safety functions",
            "evidence": "formal/",
            "status": "partial",
        }
        values.update(kwargs)
        with pytest.raises(ValueError, match=match):
            ChecklistItem(**values)


# ── SIL/ASIL Mapping Tests ──────────────────────────────────────────


class TestSILASIL:
    def test_sil_to_asil(self):
        assert SIL_TO_ASIL[SILLevel.SIL_1] == ASILLevel.ASIL_A
        assert SIL_TO_ASIL[SILLevel.SIL_4] == ASILLevel.ASIL_D


# ── CertificationGenerator Tests ────────────────────────────────────


class TestCertificationGenerator:
    def _props(self):
        return [
            FormalProperty(
                "P1",
                "sc_lif_neuron",
                "No overflow",
                "assert",
                "proven",
                sby_file="sc_lif_neuron.sby",
            ),
            FormalProperty(
                "P2", "sc_lif_neuron", "Reset", "assert", "proven", sby_file="sc_lif_neuron.sby"
            ),
            FormalProperty(
                "P3",
                "sc_bitstream_encoder",
                "Cover",
                "cover",
                "proven",
                sby_file="sc_bitstream_encoder.sby",
            ),
        ]

    def test_generate_iec(self):
        gen = CertificationGenerator()
        pkg = gen.generate(
            SafetyStandard.IEC_61508,
            SILLevel.SIL_2,
            ["sc_lif_neuron", "sc_bitstream_encoder"],
            self._props(),
        )
        assert isinstance(pkg, CertificationPackage)
        assert pkg.standard == SafetyStandard.IEC_61508
        assert pkg.package_hash != ""

    def test_generate_iso(self):
        gen = CertificationGenerator()
        pkg = gen.generate(
            SafetyStandard.ISO_26262,
            SILLevel.SIL_3,
            ["sc_lif_neuron"],
            self._props(),
        )
        assert pkg.standard == SafetyStandard.ISO_26262

    def test_generate_fda(self):
        gen = CertificationGenerator()
        pkg = gen.generate(
            SafetyStandard.FDA_CLASS_III,
            SILLevel.SIL_2,
            ["sc_lif_neuron"],
            self._props(),
        )
        assert len(pkg.checklist) == 7

    def test_traceability_in_package(self):
        gen = CertificationGenerator()
        pkg = gen.generate(
            SafetyStandard.IEC_61508,
            SILLevel.SIL_2,
            ["sc_lif_neuron"],
            self._props(),
        )
        assert "Traceability" in pkg.traceability_report

    def test_fmeda_in_package(self):
        gen = CertificationGenerator()
        pkg = gen.generate(
            SafetyStandard.IEC_61508,
            SILLevel.SIL_2,
            ["sc_lif_neuron"],
            self._props(),
        )
        assert "FMEDA" in pkg.fmeda_report

    def test_wcet_in_package(self):
        gen = CertificationGenerator()
        pkg = gen.generate(
            SafetyStandard.IEC_61508,
            SILLevel.SIL_2,
            ["sc_lif_neuron"],
            self._props(),
            {"bitstream_length": 512, "num_inputs": 16, "num_neurons": 32, "clock_mhz": 200},
        )
        assert "WCET" in pkg.wcet_report
        assert "cycles" in pkg.wcet_report

    def test_checklist_coverage(self):
        gen = CertificationGenerator()
        pkg = gen.generate(
            SafetyStandard.IEC_61508,
            SILLevel.SIL_2,
            ["sc_lif_neuron"],
            self._props(),
        )
        assert pkg.checklist_coverage > 0

    def test_checklist_coverage_rejects_corrupted_internal_state(self):
        pkg = CertificationPackage(
            standard=SafetyStandard.IEC_61508,
            sil_level=SILLevel.SIL_2,
            traceability_report="t",
            fmeda_report="f",
            formal_cert_report="p",
            wcet_report="w",
            checklist=[],
        )
        pkg.checklist.append("bad")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="ChecklistItem"):
            _ = pkg.checklist_coverage

    def test_checklist_coverage_rejects_corrupted_status_state(self):
        pkg = CertificationPackage(
            standard=SafetyStandard.IEC_61508,
            sil_level=SILLevel.SIL_2,
            traceability_report="t",
            fmeda_report="f",
            formal_cert_report="p",
            wcet_report="w",
            checklist=[ChecklistItem("id", "7.4.2", "desc", "formal/", "partial")],
        )
        pkg.checklist[0].status = "bad"  # type: ignore[assignment]
        with pytest.raises(ValueError, match="statuses"):
            _ = pkg.checklist_coverage

    def test_package_rejects_corrupted_checklist_status_state(self):
        item = ChecklistItem("id", "7.4.2", "desc", "formal/", "partial")
        item.status = "bad"  # type: ignore[assignment]
        with pytest.raises(ValueError, match="statuses"):
            CertificationPackage(
                standard=SafetyStandard.IEC_61508,
                sil_level=SILLevel.SIL_2,
                traceability_report="t",
                fmeda_report="f",
                formal_cert_report="p",
                wcet_report="w",
                checklist=[item],
            )

    def test_package_rejects_corrupted_checklist_clause_state(self):
        item = ChecklistItem("id", "7.4.2", "desc", "formal/", "partial")
        item.clause = ""  # type: ignore[assignment]
        with pytest.raises(ValueError, match="clauses"):
            CertificationPackage(
                standard=SafetyStandard.IEC_61508,
                sil_level=SILLevel.SIL_2,
                traceability_report="t",
                fmeda_report="f",
                formal_cert_report="p",
                wcet_report="w",
                checklist=[item],
            )

    def test_package_rejects_corrupted_checklist_item_id_state(self):
        item = ChecklistItem("id", "7.4.2", "desc", "formal/", "partial")
        item.item_id = ""  # type: ignore[assignment]
        with pytest.raises(ValueError, match="item_id"):
            CertificationPackage(
                standard=SafetyStandard.IEC_61508,
                sil_level=SILLevel.SIL_2,
                traceability_report="t",
                fmeda_report="f",
                formal_cert_report="p",
                wcet_report="w",
                checklist=[item],
            )

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"standard": "IEC 61508"}, "standard"),
            ({"target_sil": "SIL_2"}, "target_sil"),
            ({"modules": []}, "modules"),
            ({"modules": ["", "m2"]}, "modules"),
            ({"modules": ["sc_lif_neuron", "sc_lif_neuron"]}, "duplicates"),
            ({"modules": [" sc_lif_neuron", "m2"]}, "whitespace"),
            ({"formal_properties": "bad"}, "formal_properties"),
            ({"formal_properties": ["bad"]}, "formal_properties"),
            ({"network_config": "bad"}, "network_config"),
            ({"network_config": {"unsupported": 1}}, "network_config"),
            ({"network_config": {"bitstream_length": 0}}, "bitstream_length"),
            ({"network_config": {"num_inputs": 0}}, "num_inputs"),
            ({"network_config": {"num_neurons": 0}}, "num_neurons"),
            ({"network_config": {"clock_mhz": 0.0}}, "clock_mhz"),
        ],
    )
    def test_generate_rejects_invalid_contracts(self, kwargs, match):
        gen = CertificationGenerator()
        values = {
            "standard": SafetyStandard.IEC_61508,
            "target_sil": SILLevel.SIL_2,
            "modules": ["sc_lif_neuron"],
            "formal_properties": self._props(),
            "network_config": None,
        }
        values.update(kwargs)
        with pytest.raises(ValueError, match=match):
            gen.generate(**values)

    def test_generate_rejects_corrupted_formal_property_module_state(self):
        gen = CertificationGenerator()
        prop = FormalProperty("P1", "sc_lif_neuron", "d", "assert", "proven")
        prop.module = ""  # type: ignore[assignment]
        with pytest.raises(ValueError, match="formal_properties modules"):
            gen.generate(SafetyStandard.IEC_61508, SILLevel.SIL_2, ["sc_lif_neuron"], [prop])

    def test_generate_rejects_corrupted_formal_property_id_state(self):
        gen = CertificationGenerator()
        prop = FormalProperty("P1", "sc_lif_neuron", "d", "assert", "proven")
        prop.prop_id = ""  # type: ignore[assignment]
        with pytest.raises(ValueError, match="formal_properties prop_id"):
            gen.generate(SafetyStandard.IEC_61508, SILLevel.SIL_2, ["sc_lif_neuron"], [prop])

    def test_generate_rejects_formal_property_whitespace_state(self):
        gen = CertificationGenerator()
        prop = FormalProperty("P1", "sc_lif_neuron", "d", "assert", "proven")
        prop.module = " sc_lif_neuron"  # type: ignore[assignment]
        with pytest.raises(ValueError, match="whitespace"):
            gen.generate(SafetyStandard.IEC_61508, SILLevel.SIL_2, ["sc_lif_neuron"], [prop])

    def test_generate_rejects_corrupted_formal_property_status_state(self):
        gen = CertificationGenerator()
        prop = FormalProperty("P1", "sc_lif_neuron", "d", "assert", "proven")
        prop.status = "bad"  # type: ignore[assignment]
        with pytest.raises(ValueError, match="formal_properties statuses"):
            gen.generate(SafetyStandard.IEC_61508, SILLevel.SIL_2, ["sc_lif_neuron"], [prop])

    def test_generate_rejects_corrupted_formal_property_type_state(self):
        gen = CertificationGenerator()
        prop = FormalProperty("P1", "sc_lif_neuron", "d", "assert", "proven")
        prop.property_type = "bad"  # type: ignore[assignment]
        with pytest.raises(ValueError, match="formal_properties property_type"):
            gen.generate(SafetyStandard.IEC_61508, SILLevel.SIL_2, ["sc_lif_neuron"], [prop])

    def test_generate_rejects_duplicate_formal_property_ids(self):
        gen = CertificationGenerator()
        with pytest.raises(ValueError, match="duplicate prop_id"):
            gen.generate(
                SafetyStandard.IEC_61508,
                SILLevel.SIL_2,
                ["sc_lif_neuron"],
                [
                    FormalProperty("P1", "sc_lif_neuron", "d1", "assert", "proven"),
                    FormalProperty("P1", "sc_lif_neuron", "d2", "cover", "proven"),
                ],
            )

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"standard": "IEC 61508"}, "standard"),
            ({"sil_level": 2}, "sil_level"),
            ({"traceability_report": None}, "traceability_report"),
            ({"checklist": ["not-item"]}, "checklist"),
            ({"generated": None}, "generated"),
        ],
    )
    def test_certification_package_rejects_invalid_contracts(self, kwargs, match):
        values = {
            "standard": SafetyStandard.IEC_61508,
            "sil_level": SILLevel.SIL_2,
            "traceability_report": "trace",
            "fmeda_report": "fmeda",
            "formal_cert_report": "formal",
            "wcet_report": "wcet",
            "checklist": [
                ChecklistItem(
                    item_id="IEC 61508_7.4.2",
                    clause="7.4.2",
                    description="Formal verification of safety functions",
                    evidence="formal/",
                    status="partial",
                )
            ],
            "package_hash": "",
            "generated": "",
        }
        values.update(kwargs)
        with pytest.raises(ValueError, match=match):
            CertificationPackage(**values)


# ── Residual Risk Tests ─────────────────────────────────────────────


class TestResidualRisk:
    def test_residual_risk_all_safe(self):
        fmeda = FMEDA()
        fmeda.add_failure_mode(FailureMode("FM1", "x", "safe", FailureCategory.SAFE, 100.0))
        assert fmeda.residual_risk_fit == 0.0

    def test_residual_risk_undetected(self):
        fmeda = FMEDA()
        fmeda.add_failure_mode(
            FailureMode("FM1", "x", "bad", FailureCategory.DANGEROUS_UNDETECTED, 100.0)
        )
        assert fmeda.residual_risk_fit == 100.0

    def test_residual_risk_partial(self):
        fmeda = FMEDA()
        fmeda.add_failure_mode(
            FailureMode("FM1", "x", "det", FailureCategory.DANGEROUS_DETECTED, 100.0, 0.9)
        )
        assert 0 < fmeda.residual_risk_fit < 100.0


# ── Per-Component SFF Tests ─────────────────────────────────────────


class TestComponentSFF:
    def test_sff_by_component(self):
        fmeda = FMEDA()
        fmeda.add_sc_standard_modes("neuron")
        fmeda.add_sc_standard_modes("encoder")
        sff_map = fmeda.sff_by_component()
        assert "neuron" in sff_map
        assert "encoder" in sff_map
        assert 0 < sff_map["neuron"] <= 1.0

    def test_sff_single_component(self):
        fmeda = FMEDA()
        fmeda.add_sc_standard_modes("lif")
        sff_map = fmeda.sff_by_component()
        assert len(sff_map) == 1


# ── DO-254 / EN-50129 Generator Tests ───────────────────────────────


class TestAdditionalStandards:
    def _props(self):
        return [
            FormalProperty("P1", "sc_lif_neuron", "No overflow", "assert", "proven"),
        ]

    def test_generate_do254(self):
        gen = CertificationGenerator()
        pkg = gen.generate(
            SafetyStandard.DO_254,
            SILLevel.SIL_2,
            ["sc_lif_neuron"],
            self._props(),
        )
        assert len(pkg.checklist) == 6
        assert pkg.standard == SafetyStandard.DO_254

    def test_generate_en50129(self):
        gen = CertificationGenerator()
        pkg = gen.generate(
            SafetyStandard.EN_50129,
            SILLevel.SIL_3,
            ["sc_lif_neuron"],
            self._props(),
        )
        assert len(pkg.checklist) == 6
        assert pkg.standard == SafetyStandard.EN_50129


# ── CCF Analysis Tests (Gap 1) ─────────────────────────────────────────


class TestCCFAnalysis:
    def test_default_beta(self):
        ccf = CCFAnalysis()
        assert ccf.beta_factor == 0.10

    def test_beta_factor_rejects_corrupted_internal_state(self):
        ccf = CCFAnalysis()
        ccf.defences.append("bad")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="CCFDefence"):
            _ = ccf.beta_factor

    def test_mark_implemented(self):
        ccf = CCFAnalysis()
        assert ccf.mark_implemented("D1") is True
        assert ccf.implemented_count == 1
        assert ccf.beta_factor < 0.10

    def test_implemented_count_rejects_corrupted_internal_state(self):
        ccf = CCFAnalysis()
        ccf.defences.append("bad")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="CCFDefence"):
            _ = ccf.implemented_count

    def test_mark_implemented_normalises_whitespace(self):
        ccf = CCFAnalysis()
        assert ccf.mark_implemented(" D1 ") is True
        assert ccf.implemented_count == 1

    def test_all_implemented(self):
        ccf = CCFAnalysis()
        for d in ccf.defences:
            ccf.mark_implemented(d.defence_id)
        assert ccf.beta_factor < 0.02

    def test_sil_compatible(self):
        ccf = CCFAnalysis()
        assert ccf.sil_compatible(SILLevel.SIL_1) is True
        assert ccf.sil_compatible(SILLevel.SIL_3) is False  # beta=0.10 too high

    def test_sil_compatible_sil4_threshold(self):
        ccf = CCFAnalysis()
        for defence in ccf.defences:
            ccf.mark_implemented(defence.defence_id)
        assert ccf.sil_compatible(SILLevel.SIL_4) is True

    def test_sil_compatible_rejects_invalid_target_sil(self):
        ccf = CCFAnalysis()
        with pytest.raises(ValueError, match="target_sil"):
            ccf.sil_compatible("SIL_2")  # type: ignore[arg-type]

    def test_mark_invalid(self):
        ccf = CCFAnalysis()
        assert ccf.mark_implemented("NOPE") is False

    def test_init_rejects_duplicate_default_defence_ids(self, monkeypatch):
        monkeypatch.setattr(
            CCFAnalysis,
            "DEFAULT_DEFENCES",
            [
                CCFDefence("D1", "a", "separation", 0.01),
                CCFDefence("D1", "b", "diversity", 0.01),
            ],
        )
        with pytest.raises(ValueError, match="duplicate"):
            CCFAnalysis()

    def test_mark_implemented_rejects_invalid_defence_id(self):
        ccf = CCFAnalysis()
        with pytest.raises(ValueError, match="defence_id"):
            ccf.mark_implemented("")

    def test_mark_implemented_rejects_corrupted_internal_state(self):
        ccf = CCFAnalysis()
        ccf.defences.insert(0, "bad")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="CCFDefence"):
            ccf.mark_implemented("D1")

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"defence_id": ""}, "defence_id"),
            ({"description": ""}, "description"),
            ({"category": "other"}, "category"),
            ({"beta_reduction": -0.1}, "beta_reduction"),
            ({"beta_reduction": 1.1}, "beta_reduction"),
            ({"beta_reduction": float("nan")}, "beta_reduction"),
            ({"beta_reduction": True}, "beta_reduction"),
            ({"implemented": "yes"}, "implemented"),
        ],
    )
    def test_ccf_defence_rejects_invalid_contracts(self, kwargs, match):
        values = {
            "defence_id": "D1",
            "description": "Physical separation",
            "category": "separation",
            "beta_reduction": 0.01,
            "implemented": False,
        }
        values.update(kwargs)
        with pytest.raises(ValueError, match=match):
            CCFDefence(**values)


# ── Proof-of-Test Coverage Tests (Gap 2) ──────────────────────────────


class TestProofTestCoverage:
    def test_full_coverage(self):
        props = [
            FormalProperty("P1", "m", "d", "assert", "proven"),
            FormalProperty("P2", "m", "d", "assert", "proven"),
        ]
        assert ProofTestCoverage.coverage_from_proofs(props) == 1.0

    def test_partial_coverage(self):
        props = [
            FormalProperty("P1", "m", "d", "assert", "proven"),
            FormalProperty("P2", "m", "d", "assert", "failed"),
        ]
        assert abs(ProofTestCoverage.coverage_from_proofs(props) - 0.5) < 0.01

    def test_uncovered_modules(self):
        props = [FormalProperty("P1", "neuron", "d", "assert", "proven")]
        uncovered = ProofTestCoverage.uncovered_modules(props, ["neuron", "encoder"])
        assert uncovered == ["encoder"]

    def test_uncovered_modules_deduplicates_preserving_order(self):
        props = [FormalProperty("P1", "neuron", "d", "assert", "proven")]
        uncovered = ProofTestCoverage.uncovered_modules(props, ["encoder", "encoder", "decoder"])
        assert uncovered == ["encoder", "decoder"]

    def test_dc_to_sil(self):
        assert ProofTestCoverage.dc_to_sil(0.99).value >= 3
        assert ProofTestCoverage.dc_to_sil(0.97) == SILLevel.SIL_3
        assert ProofTestCoverage.dc_to_sil(0.50) == SILLevel.SIL_1

    @pytest.mark.parametrize("dc", [-0.1, 1.1, float("nan"), float("inf"), True, "0.9"])
    def test_dc_to_sil_rejects_invalid_contracts(self, dc):
        with pytest.raises(ValueError, match="dc"):
            ProofTestCoverage.dc_to_sil(dc)  # type: ignore[arg-type]

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
    def test_uncovered_modules_rejects_invalid_contracts(self, props, modules, match):
        with pytest.raises(ValueError, match=match):
            ProofTestCoverage.uncovered_modules(props, modules)  # type: ignore[arg-type]

    def test_uncovered_modules_rejects_corrupted_property_module(self):
        prop = FormalProperty("P1", "m", "d", "assert", "proven")
        prop.module = ""  # type: ignore[assignment]
        with pytest.raises(ValueError, match="modules"):
            ProofTestCoverage.uncovered_modules([prop], ["m"])

    def test_uncovered_modules_rejects_corrupted_property_id(self):
        prop = FormalProperty("P1", "m", "d", "assert", "proven")
        prop.prop_id = ""  # type: ignore[assignment]
        with pytest.raises(ValueError, match="prop_id"):
            ProofTestCoverage.uncovered_modules([prop], ["m"])

    @pytest.mark.parametrize(
        "props", ["invalid", [FormalProperty("P1", "n", "d", "assert", "proven"), "bad"]]
    )
    def test_coverage_from_proofs_rejects_invalid_contracts(self, props):
        with pytest.raises(ValueError, match="properties"):
            ProofTestCoverage.coverage_from_proofs(props)  # type: ignore[arg-type]

    def test_coverage_from_proofs_rejects_corrupted_property_status(self):
        prop = FormalProperty("P1", "m", "d", "assert", "proven")
        prop.status = "bad"  # type: ignore[assignment]
        with pytest.raises(ValueError, match="statuses"):
            ProofTestCoverage.coverage_from_proofs([prop])

    def test_coverage_from_proofs_rejects_corrupted_property_type(self):
        prop = FormalProperty("P1", "m", "d", "assert", "proven")
        prop.property_type = "bad"  # type: ignore[assignment]
        with pytest.raises(ValueError, match="property_type"):
            ProofTestCoverage.coverage_from_proofs([prop])

    def test_coverage_from_proofs_rejects_corrupted_property_id(self):
        prop = FormalProperty("P1", "m", "d", "assert", "proven")
        prop.prop_id = ""  # type: ignore[assignment]
        with pytest.raises(ValueError, match="prop_id"):
            ProofTestCoverage.coverage_from_proofs([prop])


# ── HFT Assessment Tests (Gap 3) ──────────────────────────────────────


class TestHFTAssessment:
    def test_high_sff_low_sil(self):
        hft = HFTAssessment(sff=0.99, target_sil=SILLevel.SIL_2)
        assert hft.required_hft == HFTLevel.HFT_0
        assert hft.is_simplex_ok

    def test_low_sff_high_sil(self):
        hft = HFTAssessment(sff=0.50, target_sil=SILLevel.SIL_3)
        assert hft.required_hft == HFTLevel.HFT_2
        assert not hft.is_simplex_ok

    def test_mid_sff(self):
        hft = HFTAssessment(sff=0.92, target_sil=SILLevel.SIL_3)
        assert hft.required_hft == HFTLevel.HFT_1

    def test_required_hft_rejects_corrupted_target_sil(self):
        hft = HFTAssessment(sff=0.92, target_sil=SILLevel.SIL_3)
        hft.target_sil = "SIL_3"  # type: ignore[assignment]
        with pytest.raises(ValueError, match="target_sil"):
            _ = hft.required_hft

    def test_required_hft_rejects_corrupted_sff_state(self):
        hft = HFTAssessment(sff=0.92, target_sil=SILLevel.SIL_3)
        hft.sff = float("nan")  # type: ignore[assignment]
        with pytest.raises(ValueError, match="sff"):
            _ = hft.required_hft

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"sff": -0.1}, "sff"),
            ({"sff": 1.1}, "sff"),
            ({"sff": float("nan")}, "sff"),
            ({"sff": float("inf")}, "sff"),
            ({"sff": True}, "sff"),
            ({"target_sil": "SIL_2"}, "target_sil"),
        ],
    )
    def test_hft_assessment_rejects_invalid_contracts(self, kwargs, match):
        values = {
            "sff": 0.90,
            "target_sil": SILLevel.SIL_2,
        }
        values.update(kwargs)
        with pytest.raises(ValueError, match=match):
            HFTAssessment(**values)


# ── Change Impact Tests (Gap 4) ───────────────────────────────────────


class TestChangeImpactTracker:
    def test_add_low_risk(self):
        ct = ChangeImpactTracker()
        ct.add_change(ChangeRecord("C1", "fix typo", ["neuron"], ["R1"], "low"))
        assert not ct.needs_re_certification

    def test_high_risk_triggers_recert(self):
        ct = ChangeImpactTracker()
        ct.add_change(ChangeRecord("C1", "redesign LIF", ["neuron"], ["R1"], "high"))
        assert ct.needs_re_certification
        assert ct.high_risk_count == 1

    def test_high_risk_count_rejects_corrupted_internal_state(self):
        ct = ChangeImpactTracker()
        ct.changes.append("bad")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="ChangeRecord"):
            _ = ct.high_risk_count

    def test_high_risk_count_rejects_corrupted_risk_level_state(self):
        ct = ChangeImpactTracker()
        change = ChangeRecord("C1", "desc", ["neuron"], ["R1"], "low")
        change.risk_level = "bad"  # type: ignore[assignment]
        ct.add_change(change)
        with pytest.raises(ValueError, match="risk_level"):
            _ = ct.high_risk_count

    def test_affected_reqs(self):
        ct = ChangeImpactTracker()
        ct.add_change(ChangeRecord("C1", "a", [], ["R1", "R2"]))
        ct.add_change(ChangeRecord("C2", "b", [], ["R2", "R3"]))
        assert ct.affected_requirements() == ["R1", "R2", "R3"]

    def test_affected_requirements_rejects_corrupted_internal_state(self):
        ct = ChangeImpactTracker()
        ct.changes.append("bad")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="ChangeRecord"):
            ct.affected_requirements()

    def test_affected_requirements_rejects_corrupted_requirement_ids(self):
        ct = ChangeImpactTracker()
        change = ChangeRecord("C1", "desc", ["neuron"], ["R1"], "low")
        change.affected_reqs = ["R1", ""]  # type: ignore[assignment]
        ct.add_change(change)
        with pytest.raises(ValueError, match="affected_reqs"):
            ct.affected_requirements()

    def test_affected_requirements_rejects_corrupted_requirement_container(self):
        ct = ChangeImpactTracker()
        change = ChangeRecord("C1", "desc", ["neuron"], ["R1"], "low")
        change.affected_reqs = "R1"  # type: ignore[assignment]
        ct.add_change(change)
        with pytest.raises(ValueError, match="affected_reqs"):
            ct.affected_requirements()

    def test_affected_requirements_rejects_corrupted_change_id_state(self):
        ct = ChangeImpactTracker()
        change = ChangeRecord("C1", "desc", ["neuron"], ["R1"], "low")
        change.change_id = ""  # type: ignore[assignment]
        ct.add_change(change)
        with pytest.raises(ValueError, match="change_id"):
            ct.affected_requirements()

    def test_add_change_rejects_invalid_contract(self):
        ct = ChangeImpactTracker()
        with pytest.raises(ValueError, match="change"):
            ct.add_change("bad")  # type: ignore[arg-type]

    def test_add_change_rejects_duplicate_change_ids(self):
        ct = ChangeImpactTracker()
        ct.add_change(ChangeRecord("C1", "a", ["n"], ["R1"], "low"))
        with pytest.raises(ValueError, match="unique"):
            ct.add_change(ChangeRecord("C1", "b", ["n"], ["R2"], "medium"))

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"change_id": ""}, "change_id"),
            ({"description": ""}, "description"),
            ({"risk_level": "critical"}, "risk_level"),
            ({"re_verification_needed": "yes"}, "re_verification_needed"),
            ({"affected_modules": "mod1"}, "affected_modules"),
            ({"affected_modules": ["", "mod"]}, "affected_modules"),
            ({"affected_reqs": "R1"}, "affected_reqs"),
            ({"affected_reqs": ["", "R1"]}, "affected_reqs"),
        ],
    )
    def test_change_record_rejects_invalid_contracts(self, kwargs, match):
        values = {
            "change_id": "C1",
            "description": "desc",
            "affected_modules": ["mod1"],
            "affected_reqs": ["R1"],
            "risk_level": "low",
            "re_verification_needed": False,
        }
        values.update(kwargs)
        with pytest.raises(ValueError, match=match):
            ChangeRecord(**values)


# ── Safety Manual Tests (Gap 5) ───────────────────────────────────────


class TestSafetyManual:
    def test_generates(self):
        manual = SafetyManualGenerator.generate(
            "SC-NeuroCore",
            SILLevel.SIL_2,
            ["sc_lif_neuron", "sc_encoder"],
            2830.0,
        )
        assert "Safety Manual" in manual
        assert "SIL 2" in manual
        assert "sc_lif_neuron" in manual
        assert "2830.0" in manual

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"product_name": ""}, "product_name"),
            ({"sil_level": "SIL_2"}, "sil_level"),
            ({"modules": []}, "modules"),
            ({"modules": ["", "m2"]}, "modules"),
            ({"modules": ["m1", "m1"]}, "duplicates"),
            ({"modules": [" m1", "m2"]}, "whitespace"),
            ({"wcet_ns": -1.0}, "wcet_ns"),
            ({"wcet_ns": float("nan")}, "wcet_ns"),
            ({"wcet_ns": True}, "wcet_ns"),
        ],
    )
    def test_generate_rejects_invalid_contracts(self, kwargs, match):
        values = {
            "product_name": "SC-NeuroCore",
            "sil_level": SILLevel.SIL_2,
            "modules": ["sc_lif_neuron"],
            "wcet_ns": 100.0,
        }
        values.update(kwargs)
        with pytest.raises(ValueError, match=match):
            SafetyManualGenerator.generate(**values)


# ── IEC 62304 Tests (Gap 6) ───────────────────────────────────────────


class TestIEC62304:
    def test_from_sil_1(self):
        a = IEC62304Assessment.from_sil(SILLevel.SIL_1)
        assert a.sw_class == SWClass.CLASS_A
        assert not a.requires_unit_testing

    def test_from_sil_3(self):
        a = IEC62304Assessment.from_sil(SILLevel.SIL_3)
        assert a.sw_class == SWClass.CLASS_C
        assert a.requires_unit_testing
        assert a.requires_architectural_design

    def test_class_b(self):
        a = IEC62304Assessment(sw_class=SWClass.CLASS_B)
        assert a.requires_unit_testing
        assert not a.requires_architectural_design

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"sw_class": "B"}, "sw_class"),
            ({"hazard_description": None}, "hazard_description"),
            ({"risk_control_measures": "measure"}, "risk_control_measures"),
            ({"risk_control_measures": ["", "measure"]}, "risk_control_measures"),
        ],
    )
    def test_iec62304_rejects_invalid_contracts(self, kwargs, match):
        values = {
            "sw_class": SWClass.CLASS_B,
            "hazard_description": "hazard",
            "risk_control_measures": ["measure 1"],
        }
        values.update(kwargs)
        with pytest.raises(ValueError, match=match):
            IEC62304Assessment(**values)


# ── Reliability / MTBF Tests (Gap 7) ──────────────────────────────────


class TestReliabilityMetrics:
    def test_mtbf(self):
        rm = ReliabilityMetrics(total_fit=100.0, dangerous_undetected_fit=5.0)
        assert rm.mtbf_hours > 0
        assert rm.mtbf_years > 0

    def test_pfh_d(self):
        rm = ReliabilityMetrics(total_fit=100.0, dangerous_undetected_fit=5.0)
        assert rm.pfh_d > 0

    def test_pfh_sil(self):
        rm = ReliabilityMetrics(total_fit=100.0, dangerous_undetected_fit=5.0)
        assert rm.pfh_sil.value >= 1

    def test_pfh_sil_rejects_corrupted_pfh_state(self):
        rm = ReliabilityMetrics(total_fit=100.0, dangerous_undetected_fit=5.0)
        rm.dangerous_undetected_fit = float("nan")  # type: ignore[assignment]
        with pytest.raises(ValueError, match="pfh_d"):
            _ = rm.pfh_sil

    def test_from_fmeda(self):
        fmeda = FMEDA()
        fmeda.add_sc_standard_modes("neuron")
        rm = ReliabilityMetrics.from_fmeda(fmeda)
        assert rm.total_fit > 0
        assert rm.mtbf_years > 0

    def test_from_fmeda_rejects_invalid_input(self):
        with pytest.raises(ValueError, match="fmeda"):
            ReliabilityMetrics.from_fmeda("bad")  # type: ignore[arg-type]

    def test_zero_fit(self):
        rm = ReliabilityMetrics(total_fit=0.0, dangerous_undetected_fit=0.0)
        assert rm.mtbf_hours == float("inf")
        assert rm.pfh_d == 0.0

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"total_fit": -1.0}, "total_fit"),
            ({"total_fit": float("nan")}, "total_fit"),
            ({"total_fit": True}, "total_fit"),
            ({"dangerous_undetected_fit": -0.1}, "dangerous_undetected_fit"),
            ({"total_fit": 1.0, "dangerous_undetected_fit": 2.0}, "cannot exceed"),
            ({"dangerous_undetected_fit": float("inf")}, "dangerous_undetected_fit"),
            ({"dangerous_undetected_fit": False}, "dangerous_undetected_fit"),
        ],
    )
    def test_reliability_metrics_reject_invalid_contracts(self, kwargs, match):
        values = {
            "total_fit": 100.0,
            "dangerous_undetected_fit": 5.0,
        }
        values.update(kwargs)
        with pytest.raises(ValueError, match=match):
            ReliabilityMetrics(**values)


# ── Evidence Bag Tests (Gap 8) ─────────────────────────────────────────


class TestEvidenceBag:
    def test_add_items(self):
        bag = EvidenceBag()
        bag.add(EvidenceItem("test.md", "report", "test"))
        assert bag.file_count == 1

    def test_file_count_rejects_corrupted_internal_state(self):
        bag = EvidenceBag()
        bag.items.append("bad")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="EvidenceItem"):
            _ = bag.file_count

    def test_from_package(self):
        gen = CertificationGenerator()
        props = [FormalProperty("P1", "m", "d", "assert", "proven")]
        pkg = gen.generate(SafetyStandard.IEC_61508, SILLevel.SIL_2, ["m"], props)
        bag = EvidenceBag()
        bag.add_from_package(pkg)
        assert bag.file_count == 5

    def test_manifest(self):
        bag = EvidenceBag()
        bag.add(EvidenceItem("x.md", "formal", "proof"))
        m = bag.manifest()
        assert "Evidence Bag" in m
        assert "x.md" in m

    def test_manifest_rejects_corrupted_internal_state(self):
        bag = EvidenceBag()
        bag.items.append("bad")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="EvidenceItem"):
            bag.manifest()

    def test_hash(self):
        bag = EvidenceBag()
        bag.add(EvidenceItem("x.md", "formal", "proof"))
        assert len(bag.compute_hashes()) == 32

    def test_hash_rejects_corrupted_internal_state(self):
        bag = EvidenceBag()
        bag.items.append("bad")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="EvidenceItem"):
            bag.compute_hashes()

    def test_hash_rejects_corrupted_duplicate_filenames_state(self):
        bag = EvidenceBag()
        bag.items = [  # type: ignore[assignment]
            EvidenceItem("x.md", "formal", "a"),
            EvidenceItem("x.md", "report", "b"),
        ]
        with pytest.raises(ValueError, match="unique"):
            bag.compute_hashes()

    def test_hash_changes_with_declared_sha256(self):
        bag_a = EvidenceBag()
        bag_a.add(EvidenceItem("x.md", "formal", "proof", sha256="a" * 64))
        bag_b = EvidenceBag()
        bag_b.add(EvidenceItem("x.md", "formal", "proof", sha256="b" * 64))
        assert bag_a.compute_hashes() != bag_b.compute_hashes()

    def test_add_rejects_invalid_item(self):
        bag = EvidenceBag()
        with pytest.raises(ValueError, match="item"):
            bag.add("bad")  # type: ignore[arg-type]

    def test_add_rejects_duplicate_filenames(self):
        bag = EvidenceBag()
        bag.add(EvidenceItem("x.md", "formal", "proof"))
        with pytest.raises(ValueError, match="unique"):
            bag.add(EvidenceItem("x.md", "report", "duplicate"))

    def test_add_from_package_rejects_invalid_package(self):
        bag = EvidenceBag()
        with pytest.raises(ValueError, match="pkg"):
            bag.add_from_package("bad")  # type: ignore[arg-type]

    def test_add_from_package_rejects_corrupted_package_checklist_state(self):
        gen = CertificationGenerator()
        pkg = gen.generate(
            SafetyStandard.IEC_61508,
            SILLevel.SIL_2,
            ["sc_lif_neuron"],
            [FormalProperty("P1", "sc_lif_neuron", "d", "assert", "proven")],
        )
        pkg.checklist.append("bad")  # type: ignore[arg-type]
        bag = EvidenceBag()
        with pytest.raises(ValueError, match="ChecklistItem"):
            bag.add_from_package(pkg)

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"filename": ""}, "filename"),
            ({"category": "unsafe"}, "category"),
            ({"description": ""}, "description"),
            ({"sha256": None}, "sha256"),
            ({"sha256": "not_hex"}, "hexadecimal"),
        ],
    )
    def test_evidence_item_rejects_invalid_contracts(self, kwargs, match):
        values = {
            "filename": "formal_proof_cert.md",
            "category": "formal",
            "description": "Formal proof certificate",
            "sha256": "",
        }
        values.update(kwargs)
        with pytest.raises(ValueError, match=match):
            EvidenceItem(**values)


# ── Cross-Standard Mapping Tests (Gap 9) ──────────────────────────────


class TestCrossStandardMapper:
    def test_equivalent_clauses(self):
        equiv = CrossStandardMapper.equivalent_clauses("IEC 61508", "7.4.2")
        assert len(equiv) == 2
        assert ("ISO 26262", "6.7.4") in equiv

    def test_equivalent_clauses_normalises_whitespace(self):
        equiv = CrossStandardMapper.equivalent_clauses(" IEC 61508 ", " 7.4.2 ")
        assert ("ISO 26262", "6.7.4") in equiv

    def test_no_mapping(self):
        equiv = CrossStandardMapper.equivalent_clauses("IEC 61508", "99.99")
        assert equiv == []

    def test_coverage_overlap_rejects_malformed_item_id(self):
        bad_item = ChecklistItem(
            item_id="MALFORMED",
            clause="7.4.2",
            description="desc",
            evidence="formal/",
            status="partial",
        )
        good_item = ChecklistItem(
            item_id="ISO 26262_6.7.4",
            clause="6.7.4",
            description="desc",
            evidence="formal/",
            status="partial",
        )
        with pytest.raises(ValueError, match="item_id"):
            CrossStandardMapper.coverage_overlap([bad_item], [good_item])

    def test_coverage_overlap_rejects_malformed_right_item_id(self):
        good_item = ChecklistItem(
            item_id="IEC 61508_7.4.2",
            clause="7.4.2",
            description="desc",
            evidence="formal/",
            status="partial",
        )
        bad_item = ChecklistItem(
            item_id="MALFORMED",
            clause="6.7.4",
            description="desc",
            evidence="formal/",
            status="partial",
        )
        with pytest.raises(ValueError, match="checklist_b"):
            CrossStandardMapper.coverage_overlap([good_item], [bad_item])

    def test_coverage_overlap_rejects_corrupted_clause_state(self):
        left = ChecklistItem(
            item_id="IEC 61508_7.4.2",
            clause="7.4.2",
            description="desc",
            evidence="formal/",
            status="partial",
        )
        right = ChecklistItem(
            item_id="ISO 26262_6.7.4",
            clause="6.7.4",
            description="desc",
            evidence="formal/",
            status="partial",
        )
        right.clause = ""  # type: ignore[assignment]
        with pytest.raises(ValueError, match="clauses"):
            CrossStandardMapper.coverage_overlap([left], [right])

    def test_coverage_overlap_rejects_corrupted_status_state(self):
        left = ChecklistItem("IEC 61508_7.4.2", "7.4.2", "desc", "formal/", "partial")
        right = ChecklistItem("ISO 26262_6.7.4", "6.7.4", "desc", "formal/", "partial")
        left.status = "bad"  # type: ignore[assignment]
        with pytest.raises(ValueError, match="statuses"):
            CrossStandardMapper.coverage_overlap([left], [right])

    def test_coverage_overlap_deduplicates_equivalent_mappings(self):
        left = [
            ChecklistItem("IEC 61508_7.4.2", "7.4.2", "desc", "formal/", "partial"),
            ChecklistItem("IEC 61508_7.4.2_b", "7.4.2", "desc2", "formal/", "partial"),
        ]
        right = [ChecklistItem("ISO 26262_6.7.4", "6.7.4", "desc", "formal/", "partial")]
        assert CrossStandardMapper.coverage_overlap(left, right) == 1

    @pytest.mark.parametrize(
        ("standard", "clause", "match"),
        [
            ("", "7.4.2", "standard"),
            ("IEC 61508", "", "clause"),
        ],
    )
    def test_equivalent_clauses_rejects_invalid_contracts(self, standard, clause, match):
        with pytest.raises(ValueError, match=match):
            CrossStandardMapper.equivalent_clauses(standard, clause)

    def test_equivalent_clauses_rejects_corrupted_mapping_state(self):
        key = ("IEC 61508", "7.4.2")
        original = CROSS_MAP[key]
        try:
            CROSS_MAP[key] = [("ISO 26262", ""), ("DO-254", "6.0")]  # type: ignore[list-item]
            with pytest.raises(ValueError, match="mappings"):
                CrossStandardMapper.equivalent_clauses("IEC 61508", "7.4.2")
        finally:
            CROSS_MAP[key] = original

    @pytest.mark.parametrize(
        ("left", "right", "match"),
        [
            ("invalid", [], "lists"),
            ([], "invalid", "lists"),
            (["bad"], [], "checklist_a"),
            ([], ["bad"], "checklist_b"),
        ],
    )
    def test_coverage_overlap_rejects_invalid_contracts(self, left, right, match):
        with pytest.raises(ValueError, match=match):
            CrossStandardMapper.coverage_overlap(left, right)  # type: ignore[arg-type]


# ── Formal Property Gap Detector Tests (Gap 10) ───────────────────────


class TestFormalGapDetector:
    def test_fully_covered(self):
        props = [
            FormalProperty("P1", "neuron", "d", "assert", "proven"),
            FormalProperty("P2", "neuron", "d", "cover", "proven"),
        ]
        assert FormalPropertyGapDetector.is_fully_covered(props, ["neuron"])

    def test_missing_module(self):
        props = [FormalProperty("P1", "neuron", "d", "assert", "proven")]
        gaps = FormalPropertyGapDetector.detect(props, ["neuron", "encoder"])
        assert len(gaps) >= 1
        assert any(g.module == "encoder" for g in gaps)

    def test_detect_deduplicates_required_modules(self):
        props = [FormalProperty("P1", "neuron", "d", "assert", "proven")]
        gaps = FormalPropertyGapDetector.detect(props, ["encoder", "encoder"])
        assert [g.module for g in gaps] == ["encoder"]

    def test_failed_property(self):
        props = [
            FormalProperty("P1", "neuron", "d", "assert", "failed"),
            FormalProperty("P2", "neuron", "d", "cover", "proven"),
        ]
        gaps = FormalPropertyGapDetector.detect(props, ["neuron"])
        assert len(gaps) == 1
        assert gaps[0].proven_properties == 1

    def test_gap_coverage(self):
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
    def test_detect_rejects_invalid_contracts(self, properties, required_modules, match):
        with pytest.raises(ValueError, match=match):
            FormalPropertyGapDetector.detect(properties, required_modules)  # type: ignore[arg-type]

    def test_detect_rejects_corrupted_property_type_state(self):
        prop = FormalProperty("P1", "neuron", "d", "assert", "proven")
        prop.property_type = "bad"  # type: ignore[assignment]
        with pytest.raises(ValueError, match="property_type"):
            FormalPropertyGapDetector.detect([prop], ["neuron"])

    def test_detect_rejects_corrupted_property_status_state(self):
        prop = FormalProperty("P1", "neuron", "d", "assert", "proven")
        prop.status = "bad"  # type: ignore[assignment]
        with pytest.raises(ValueError, match="statuses"):
            FormalPropertyGapDetector.detect([prop], ["neuron"])

    def test_detect_rejects_corrupted_property_module_state(self):
        prop = FormalProperty("P1", "neuron", "d", "assert", "proven")
        prop.module = ""  # type: ignore[assignment]
        with pytest.raises(ValueError, match="modules"):
            FormalPropertyGapDetector.detect([prop], ["neuron"])

    def test_detect_rejects_corrupted_property_id_state(self):
        prop = FormalProperty("P1", "neuron", "d", "assert", "proven")
        prop.prop_id = ""  # type: ignore[assignment]
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
    def test_property_gap_rejects_invalid_contracts(self, kwargs, match):
        values = {
            "module": "neuron",
            "total_properties": 2,
            "proven_properties": 1,
            "missing_types": ["assert"],
        }
        values.update(kwargs)
        with pytest.raises(ValueError, match=match):
            PropertyGap(**values)


class TestSafetyCertificationBoundaryContracts:
    def test_traceability_unknown_verification_link_is_rejected_without_mutation(self):
        tm = TraceabilityMatrix()
        assert tm.link_verification("REQ_MISSING", "formal/proof.sby") is False
        assert tm.requirements == {}

    def test_traceability_rejects_corrupted_verification_reference_state(self):
        tm = TraceabilityMatrix()
        req = Requirement("REQ_001", "Test", SafetyStandard.IEC_61508)
        tm.add_requirement(req)
        req.verification_refs.append("")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="verification_refs"):
            tm.link_implementation("REQ_001", "hdl/test.v")

    def test_traceability_empty_matrix_reports_zero_fraction(self):
        assert TraceabilityMatrix().coverage == 0.0

    def test_fmeda_rejects_nonfinite_aggregate_total(self):
        fmeda = FMEDA()
        fmeda.add_failure_mode(FailureMode("F1", "core", "desc", FailureCategory.SAFE, 1e308, 1.0))
        fmeda.add_failure_mode(FailureMode("F2", "core", "desc", FailureCategory.SAFE, 1e308, 1.0))
        with pytest.raises(ValueError, match="total_failure_rate"):
            _ = fmeda.total_failure_rate

    def test_fmeda_empty_and_zero_rate_paths_are_bounded(self):
        empty = FMEDA()
        assert empty.safe_failure_fraction == 0.0

        zero = FMEDA()
        zero.add_failure_mode(FailureMode("F1", "core", "desc", FailureCategory.SAFE, 0.0, 1.0))
        assert zero.safe_failure_fraction == 0.0

        detected_zero = FMEDA()
        detected_zero.add_failure_mode(
            FailureMode("F2", "core", "desc", FailureCategory.DANGEROUS_DETECTED, 0.0, 0.9)
        )
        with pytest.raises(ValueError, match="denominator"):
            _ = detected_zero.diagnostic_coverage

    def test_fmeda_component_sff_rejects_nonfinite_component_total(self):
        fmeda = FMEDA()
        fmeda.add_failure_mode(FailureMode("F1", "core", "desc", FailureCategory.SAFE, 1e308, 1.0))
        fmeda.add_failure_mode(FailureMode("F2", "core", "desc", FailureCategory.SAFE, 1e308, 1.0))
        with pytest.raises(ValueError, match="component failure-rate totals"):
            fmeda.sff_by_component()

    def test_fmeda_sil_threshold_boundaries(self, monkeypatch):
        sil4 = FMEDA()
        sil4.add_failure_mode(
            FailureMode("F1", "core", "desc", FailureCategory.DANGEROUS_DETECTED, 1.0, 0.99)
        )
        assert sil4.max_achievable_sil() == SILLevel.SIL_4

        sil3 = FMEDA()
        sil3.add_failure_mode(
            FailureMode("F2", "core", "desc", FailureCategory.DANGEROUS_DETECTED, 1.0, 0.99)
        )
        sil3.add_failure_mode(
            FailureMode("F3", "core", "desc", FailureCategory.DANGEROUS_UNDETECTED, 0.0102, 0.0)
        )
        assert sil3.max_achievable_sil() == SILLevel.SIL_3

        sil1 = FMEDA()
        sil1.add_failure_mode(
            FailureMode("F4", "core", "desc", FailureCategory.DANGEROUS_UNDETECTED, 1.0, 0.0)
        )
        assert sil1.max_achievable_sil() == SILLevel.SIL_1

        monkeypatch.setattr(FMEDA, "safe_failure_fraction", property(lambda self: float("nan")))
        with pytest.raises(ValueError, match="safe_failure_fraction"):
            sil4.max_achievable_sil()

    def test_formal_certificate_add_property_and_report_validation(self):
        cert = FormalProofCertificate()
        prop = FormalProperty("P1", "neuron", "desc", "assert", "proven")
        cert.add_property(prop)
        assert cert.total_count == 1

        prop.property_type = "invalid"  # type: ignore[assignment]
        with pytest.raises(ValueError, match="property_type"):
            cert.generate_report()

    def test_empty_checklist_package_reports_zero_fraction(self):
        pkg = CertificationPackage(
            standard=SafetyStandard.IEC_61508,
            sil_level=SILLevel.SIL_2,
            traceability_report="trace",
            fmeda_report="fmeda",
            formal_cert_report="formal",
            wcet_report="wcet",
            checklist=[],
        )
        assert pkg.checklist_coverage == 0.0

    def test_package_revalidates_clause_after_status_contract(self):
        class FlakyClause(str):
            calls = 0

            def strip(self, chars=None):  # noqa: ANN001
                self.calls += 1
                return "7.4.2" if self.calls == 1 else ""

        item = ChecklistItem("IEC 61508_7.4.2", "7.4.2", "desc", "formal/", "partial")
        item.clause = FlakyClause("7.4.2")  # type: ignore[assignment]

        with pytest.raises(ValueError, match="checklist clauses"):
            CertificationPackage(
                standard=SafetyStandard.IEC_61508,
                sil_level=SILLevel.SIL_2,
                traceability_report="trace",
                fmeda_report="fmeda",
                formal_cert_report="formal",
                wcet_report="wcet",
                checklist=[item],
            )

    def test_certification_generator_rejects_formal_property_id_whitespace(self):
        prop = FormalProperty(" P1", "neuron", "desc", "assert", "proven")
        with pytest.raises(ValueError, match="prop_id"):
            CertificationGenerator().generate(
                SafetyStandard.IEC_61508,
                SILLevel.SIL_2,
                ["neuron"],
                [prop],
            )

    def test_certification_generator_rejects_boolean_clock_configuration(self):
        prop = FormalProperty("P1", "neuron", "desc", "assert", "proven")
        with pytest.raises(ValueError, match="clock_mhz"):
            CertificationGenerator().generate(
                SafetyStandard.IEC_61508,
                SILLevel.SIL_2,
                ["neuron"],
                [prop],
                network_config={"clock_mhz": True},
            )

    def test_common_cause_default_defence_contracts(self, monkeypatch):
        original = CCFAnalysis.DEFAULT_DEFENCES
        monkeypatch.setattr(CCFAnalysis, "DEFAULT_DEFENCES", ["bad"])
        with pytest.raises(ValueError, match="DEFAULT_DEFENCES"):
            CCFAnalysis()
        monkeypatch.setattr(CCFAnalysis, "DEFAULT_DEFENCES", original)

        ccf = CCFAnalysis()
        ccf.defences[0].beta_reduction = float("nan")  # type: ignore[assignment]
        with pytest.raises(ValueError, match="beta_reduction"):
            _ = ccf.beta_factor

    def test_proof_test_assessment_boundaries(self):
        assert (
            ProofTestCoverage.coverage_from_proofs(
                [FormalProperty("P1", "m", "desc", "cover", "proven")]
            )
            == 0.0
        )
        assert ProofTestCoverage.dc_to_sil(0.90) == SILLevel.SIL_2
        assert ProofTestCoverage.dc_to_sil(0.60) == SILLevel.SIL_1

    @pytest.mark.parametrize(
        ("sff", "target", "expected"),
        [
            (0.99, SILLevel.SIL_4, HFTLevel.HFT_1),
            (0.90, SILLevel.SIL_2, HFTLevel.HFT_0),
            (0.90, SILLevel.SIL_3, HFTLevel.HFT_1),
            (0.90, SILLevel.SIL_4, HFTLevel.HFT_2),
            (0.60, SILLevel.SIL_1, HFTLevel.HFT_0),
            (0.60, SILLevel.SIL_2, HFTLevel.HFT_1),
            (0.60, SILLevel.SIL_3, HFTLevel.HFT_2),
            (0.59, SILLevel.SIL_1, HFTLevel.HFT_1),
        ],
    )
    def test_hft_threshold_boundaries(self, sff, target, expected):
        assert HFTAssessment(sff=sff, target_sil=target).required_hft == expected

    @pytest.mark.parametrize(
        ("dangerous_undetected_fit", "expected"),
        [
            (10.0, SILLevel.SIL_4),
            (100.0, SILLevel.SIL_3),
            (1000.0, SILLevel.SIL_2),
            (2000.0, SILLevel.SIL_1),
        ],
    )
    def test_reliability_pfh_sil_thresholds(self, dangerous_undetected_fit, expected):
        metrics = ReliabilityMetrics(
            total_fit=dangerous_undetected_fit,
            dangerous_undetected_fit=dangerous_undetected_fit,
        )
        assert metrics.pfh_sil == expected

    def test_cross_standard_overlap_rejects_empty_clauses_on_each_side(self):
        left = ChecklistItem("IEC 61508_7.4.2", "7.4.2", "desc", "formal/", "partial")
        right = ChecklistItem("ISO 26262_6.7.4", "6.7.4", "desc", "formal/", "partial")

        left.clause = ""  # type: ignore[assignment]
        with pytest.raises(ValueError, match="checklist_a clauses"):
            CrossStandardMapper.coverage_overlap([left], [right])

        left.clause = "7.4.2"  # type: ignore[assignment]
        right.status = "invalid"  # type: ignore[assignment]
        with pytest.raises(ValueError, match="checklist_b statuses"):
            CrossStandardMapper.coverage_overlap([left], [right])

    def test_property_gap_rejects_boolean_proven_count(self):
        with pytest.raises(ValueError, match="proven_properties"):
            PropertyGap("module", 2, True, ["assert"])

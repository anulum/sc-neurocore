# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Safety Certification Generator Tests

import sys
import os


sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "..", "src", "sc_neurocore", "safety_cert")
)

from safety_cert import (
    ASILLevel,
    CCFAnalysis,
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
    WCETAnalyzer,
)


# ── TraceabilityMatrix Tests ─────────────────────────────────────────


class TestTraceabilityMatrix:
    def test_add_requirement(self):
        tm = TraceabilityMatrix()
        req = Requirement("REQ_001", "Test", SafetyStandard.IEC_61508)
        tm.add_requirement(req)
        assert "REQ_001" in tm.requirements

    def test_link_implementation(self):
        tm = TraceabilityMatrix()
        tm.add_requirement(Requirement("REQ_001", "Test", SafetyStandard.IEC_61508))
        assert tm.link_implementation("REQ_001", "hdl/test.v") is True
        assert tm.requirements["REQ_001"].status == "implemented"

    def test_link_verification(self):
        tm = TraceabilityMatrix()
        req = Requirement("REQ_001", "Test", SafetyStandard.IEC_61508)
        req.implementation_refs = ["hdl/test.v"]
        tm.add_requirement(req)
        tm.link_verification("REQ_001", "formal/test.sby")
        assert tm.requirements["REQ_001"].status == "verified"

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

    def test_open_count(self):
        tm = TraceabilityMatrix()
        tm.add_requirement(Requirement("R1", "Test", SafetyStandard.IEC_61508))
        assert tm.open_count == 1

    def test_link_nonexistent(self):
        tm = TraceabilityMatrix()
        assert tm.link_implementation("NOPE", "x.v") is False

    def test_generate_report(self):
        tm = TraceabilityMatrix()
        tm.add_requirement(Requirement("R1", "Test", SafetyStandard.IEC_61508))
        report = tm.generate_report()
        assert "Traceability Matrix" in report
        assert "R1" in report


# ── FMEDA Tests ──────────────────────────────────────────────────────


class TestFMEDA:
    def test_add_failure_mode(self):
        fmeda = FMEDA()
        fm = FailureMode("FM1", "neuron", "stuck", FailureCategory.SAFE, 10.0)
        fmeda.add_failure_mode(fm)
        assert len(fmeda.failure_modes) == 1

    def test_add_sc_standard_modes(self):
        fmeda = FMEDA()
        fmeda.add_sc_standard_modes("sc_lif_neuron")
        assert len(fmeda.failure_modes) == 5

    def test_total_failure_rate(self):
        fmeda = FMEDA()
        fmeda.add_sc_standard_modes("neuron")
        assert fmeda.total_failure_rate > 0

    def test_safe_failure_fraction(self):
        fmeda = FMEDA()
        fmeda.add_sc_standard_modes("neuron")
        sff = fmeda.safe_failure_fraction
        assert 0.0 < sff <= 1.0

    def test_diagnostic_coverage(self):
        fmeda = FMEDA()
        fmeda.add_sc_standard_modes("neuron")
        dc = fmeda.diagnostic_coverage
        assert 0.0 < dc <= 1.0

    def test_max_sil(self):
        fmeda = FMEDA()
        fmeda.add_sc_standard_modes("neuron")
        sil = fmeda.max_achievable_sil()
        assert sil.value >= 1

    def test_generate_report(self):
        fmeda = FMEDA()
        fmeda.add_sc_standard_modes("neuron")
        report = fmeda.generate_report()
        assert "FMEDA" in report
        assert "FIT" in report

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

    def test_pass_rate(self):
        cert = FormalProofCertificate(properties=self._props())
        assert abs(cert.pass_rate - 0.75) < 0.01

    def test_compute_hash(self):
        cert = FormalProofCertificate(properties=self._props())
        h = cert.compute_hash()
        assert len(h) == 32

    def test_hash_deterministic(self):
        cert = FormalProofCertificate(properties=self._props())
        assert cert.compute_hash() == cert.compute_hash()

    def test_generate_report(self):
        cert = FormalProofCertificate(properties=self._props())
        report = cert.generate_report()
        assert "Formal Proof Certificate" in report
        assert "P1" in report


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

    def test_mark_implemented(self):
        ccf = CCFAnalysis()
        assert ccf.mark_implemented("D1") is True
        assert ccf.implemented_count == 1
        assert ccf.beta_factor < 0.10

    def test_all_implemented(self):
        ccf = CCFAnalysis()
        for d in ccf.defences:
            ccf.mark_implemented(d.defence_id)
        assert ccf.beta_factor < 0.02

    def test_sil_compatible(self):
        ccf = CCFAnalysis()
        assert ccf.sil_compatible(SILLevel.SIL_1) is True
        assert ccf.sil_compatible(SILLevel.SIL_3) is False  # beta=0.10 too high

    def test_mark_invalid(self):
        ccf = CCFAnalysis()
        assert ccf.mark_implemented("NOPE") is False


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

    def test_dc_to_sil(self):
        assert ProofTestCoverage.dc_to_sil(0.99).value >= 3
        assert ProofTestCoverage.dc_to_sil(0.50) == SILLevel.SIL_1


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

    def test_affected_reqs(self):
        ct = ChangeImpactTracker()
        ct.add_change(ChangeRecord("C1", "a", [], ["R1", "R2"]))
        ct.add_change(ChangeRecord("C2", "b", [], ["R2", "R3"]))
        assert ct.affected_requirements() == ["R1", "R2", "R3"]


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

    def test_from_fmeda(self):
        fmeda = FMEDA()
        fmeda.add_sc_standard_modes("neuron")
        rm = ReliabilityMetrics.from_fmeda(fmeda)
        assert rm.total_fit > 0
        assert rm.mtbf_years > 0

    def test_zero_fit(self):
        rm = ReliabilityMetrics(total_fit=0.0, dangerous_undetected_fit=0.0)
        assert rm.mtbf_hours == float("inf")
        assert rm.pfh_d == 0.0


# ── Evidence Bag Tests (Gap 8) ─────────────────────────────────────────


class TestEvidenceBag:
    def test_add_items(self):
        bag = EvidenceBag()
        bag.add(EvidenceItem("test.md", "report", "test"))
        assert bag.file_count == 1

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

    def test_hash(self):
        bag = EvidenceBag()
        bag.add(EvidenceItem("x.md", "formal", "proof"))
        assert len(bag.compute_hashes()) == 32


# ── Cross-Standard Mapping Tests (Gap 9) ──────────────────────────────


class TestCrossStandardMapper:
    def test_equivalent_clauses(self):
        equiv = CrossStandardMapper.equivalent_clauses("IEC 61508", "7.4.2")
        assert len(equiv) == 2
        assert ("ISO 26262", "6.7.4") in equiv

    def test_no_mapping(self):
        equiv = CrossStandardMapper.equivalent_clauses("IEC 61508", "99.99")
        assert equiv == []


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

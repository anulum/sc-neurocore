# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestAdaptivePrecisionAPISurface from former test_adaptive_precision.py

"""Focused suite: TestAdaptivePrecisionAPISurface from former test_adaptive_precision.py."""

from __future__ import annotations

from tests.adaptive_precision_support import *  # noqa: F403


class TestAdaptivePrecisionAPISurface:
    """Public facade and formal-evidence bundle checks."""

    def test_auto_tune_manifest_binds_percent_target_contract(self) -> None:
        """Auto-tuning binds percent targets into the public manifest."""
        manifest = auto_tune_synapse_precisions(
            [np.array([[0.2, 0.8]])],
            target_error_percent=0.1,
            min_bits=2,
            max_bits=8,
            min_length=16,
            max_length=512,
        )

        assert manifest["schema"] == "sc-neurocore.adaptive_precision_plan.v1"
        assert manifest["api_surface"]["action_id"] == "auto_tune_adaptive_precision"
        assert manifest["api_surface"]["target_error_percent"] == 0.1
        assert manifest["api_surface"]["target_error_fraction"] == 0.001
        assert manifest["api_surface"]["objective"] == "minimal_luts_under_error_target"
        assert manifest["api_surface"]["cost_metric"] == "sum(bit_width * log2(bitstream_length))"
        assert manifest["api_surface"]["estimated_lut_cost"] > 0.0
        assert (
            manifest["api_surface"]["uniform_length_reference_cost"]
            >= manifest["api_surface"]["estimated_lut_cost"]
        )
        assert manifest["num_synapses"] == 2

    def test_auto_tune_rejects_non_positive_percent_target(self) -> None:
        """Auto-tuning rejects non-positive percent error targets."""
        with np.testing.assert_raises_regex(ValueError, "target_error_percent"):
            auto_tune_synapse_precisions([np.array([[0.2]])], target_error_percent=0.0)

    def test_write_formal_bundle_materialises_sva_sby_and_manifest(self, tmp_path: Path) -> None:
        """Formal evidence bundle writing materializes SVA, SBY, and JSON."""
        assignments = assign_synapse_precisions(
            [np.array([[0.25, 0.75]])],
            target_error=0.01,
            min_bits=2,
            max_bits=8,
            min_length=16,
            max_length=256,
        )
        manifest = write_precision_formal_evidence_bundle(
            tmp_path,
            assignments,
            module_name="adaptive_precision_plan",
        )

        assert manifest["schema_version"] == "sc-neurocore.adaptive-precision-formal-bundle.v1"
        assert manifest["formal_claim"]["symbiyosys_executed"] is False
        assert manifest["formal_claim"]["formal_proof_passed"] is False
        assert manifest["formal_claim"]["hardware_measurement_claimed"] is False
        assert manifest["evidence_boundary"] == (
            "bundle_generation_only_no_symbiyosys_execution_no_silicon_claim"
        )
        # The RTL monitor is now materialised alongside the checker (it used to be
        # named in the manifest but never written).
        assert (tmp_path / "adaptive_precision_plan.v").is_file()
        assert (tmp_path / "adaptive_precision_plan_sva.sv").is_file()
        assert (tmp_path / "adaptive_precision_plan.sby").is_file()
        manifest_path = tmp_path / "adaptive_precision_plan_formal_manifest.json"
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert payload == manifest
        # The checker carries real proof obligations, not placeholder comments.
        sva_text = (tmp_path / "adaptive_precision_plan_sva.sv").read_text(encoding="utf-8")
        assert "assert (" in sva_text
        assert "assume (" in sva_text
        # The emitted script is a complete BMC (k-induction would spuriously fail
        # the non-inductive accumulator bound).
        sby_text = (tmp_path / "adaptive_precision_plan.sby").read_text(encoding="utf-8")
        assert "mode bmc" in sby_text

    def test_write_formal_bundle_rejects_empty_assignments(self, tmp_path: Path) -> None:
        """Formal evidence bundle writing rejects empty assignment lists."""
        with np.testing.assert_raises_regex(ValueError, "assignments must not be empty"):
            write_precision_formal_evidence_bundle(tmp_path, [])

    def test_execute_records_real_verdict_when_proof_passes(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """execute=True with a passing proof records the machine-checked verdict."""
        assignments = assign_synapse_precisions(
            [np.array([[0.25, 0.75]])],
            target_error=0.05,
            min_bits=2,
            max_bits=8,
            min_length=8,
            max_length=16,
        )
        monkeypatch.setattr(
            formal_property_check, "formal_tools_available", lambda engine="z3": True
        )
        monkeypatch.setattr(
            formal_property_check,
            "prove_property",
            lambda *a, **k: PropertyProofResult(
                proven=True, verdict="PASS", mode="bmc", depth=18, engine="z3", returncode=0
            ),
        )
        manifest = write_precision_formal_evidence_bundle(tmp_path, assignments, execute=True)
        claim = manifest["formal_claim"]
        assert claim["symbiyosys_executed"] is True
        assert claim["formal_proof_passed"] is True
        assert claim["proof_verdict"] == "PASS"
        assert claim["proof_mode"] == "bmc"
        assert "proof_counterexample" not in claim
        assert manifest["evidence_boundary"] == "symbiyosys_bmc_executed_no_silicon_claim"

    def test_execute_records_disproof_with_counterexample(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """execute=True with a failing proof records the counterexample, not a pass."""
        assignments = assign_synapse_precisions(
            [np.array([[0.25, 0.75]])],
            target_error=0.05,
            min_bits=2,
            max_bits=8,
            min_length=8,
            max_length=16,
        )
        monkeypatch.setattr(
            formal_property_check, "formal_tools_available", lambda engine="z3": True
        )
        monkeypatch.setattr(
            formal_property_check,
            "prove_property",
            lambda *a, **k: PropertyProofResult(
                proven=False,
                verdict="FAIL",
                mode="bmc",
                depth=18,
                engine="z3",
                returncode=2,
                counterexample="failed assertion at step 4",
            ),
        )
        manifest = write_precision_formal_evidence_bundle(tmp_path, assignments, execute=True)
        claim = manifest["formal_claim"]
        assert claim["symbiyosys_executed"] is True
        assert claim["formal_proof_passed"] is False
        assert claim["proof_counterexample"] == "failed assertion at step 4"

    def test_execute_records_skip_when_toolchain_absent(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """execute=True without tools records a skip reason, never a fabricated pass."""
        assignments = assign_synapse_precisions(
            [np.array([[0.25, 0.75]])],
            target_error=0.05,
            min_bits=2,
            max_bits=8,
            min_length=8,
            max_length=16,
        )
        monkeypatch.setattr(
            formal_property_check, "formal_tools_available", lambda engine="z3": False
        )
        manifest = write_precision_formal_evidence_bundle(tmp_path, assignments, execute=True)
        claim = manifest["formal_claim"]
        assert claim["symbiyosys_executed"] is False
        assert claim["formal_proof_passed"] is False
        assert "sby" in claim["execution_skipped_reason"]
        assert manifest["evidence_boundary"] == (
            "bundle_generation_only_no_symbiyosys_execution_no_silicon_claim"
        )

    @_needs_formal
    def test_execute_runs_real_proof_end_to_end(self, tmp_path: Path) -> None:
        """With the toolchain present, execute=True machine-checks the real bundle."""
        assignments = assign_synapse_precisions(
            [np.array([[0.25, 0.75]])],
            target_error=0.05,
            min_bits=2,
            max_bits=8,
            min_length=8,
            max_length=16,
        )
        manifest = write_precision_formal_evidence_bundle(tmp_path, assignments, execute=True)
        claim = manifest["formal_claim"]
        assert claim["symbiyosys_executed"] is True
        assert claim["formal_proof_passed"] is True
        assert claim["proof_verdict"] == "PASS"

    def test_unbounded_emits_kinduction_script_and_records_mode(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """unbounded=True emits a k-induction script and records the unbounded proof."""
        assignments = assign_synapse_precisions(
            [np.array([[0.25, 0.75]])],
            target_error=0.05,
            min_bits=2,
            max_bits=8,
            min_length=8,
            max_length=16,
        )
        monkeypatch.setattr(
            formal_property_check, "formal_tools_available", lambda engine="z3": True
        )
        monkeypatch.setattr(
            formal_property_check,
            "prove_property",
            lambda *a, **k: PropertyProofResult(
                proven=True, verdict="PASS", mode="prove", depth=8, engine="z3", returncode=0
            ),
        )
        manifest = write_precision_formal_evidence_bundle(
            tmp_path, assignments, execute=True, unbounded=True
        )
        claim = manifest["formal_claim"]
        assert claim["formal_proof_passed"] is True
        assert claim["proof_mode"] == "prove"
        assert claim["proof_unbounded"] is True
        assert manifest["evidence_boundary"] == "symbiyosys_kinduction_executed_no_silicon_claim"
        assert "mode prove" in (tmp_path / "adaptive_precision_plan.sby").read_text(
            encoding="utf-8"
        )

    def test_unbounded_records_inconclusive_without_fabricating_pass(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """An inconclusive k-induction is recorded as unproven, not a pass."""
        assignments = assign_synapse_precisions(
            [np.array([[0.25, 0.75]])],
            target_error=0.05,
            min_bits=2,
            max_bits=8,
            min_length=8,
            max_length=16,
        )
        monkeypatch.setattr(
            formal_property_check, "formal_tools_available", lambda engine="z3": True
        )
        monkeypatch.setattr(
            formal_property_check,
            "prove_property",
            lambda *a, **k: PropertyProofResult(
                proven=False, verdict="UNKNOWN", mode="prove", depth=8, engine="z3", returncode=4
            ),
        )
        manifest = write_precision_formal_evidence_bundle(
            tmp_path, assignments, execute=True, unbounded=True
        )
        claim = manifest["formal_claim"]
        assert claim["symbiyosys_executed"] is True
        assert claim["formal_proof_passed"] is False
        assert claim["proof_verdict"] == "UNKNOWN"
        assert "converge" in claim["proof_inconclusive_reason"]
        assert "proof_counterexample" not in claim

    @_needs_formal
    def test_unbounded_runs_real_kinduction_end_to_end(self, tmp_path: Path) -> None:
        """With the toolchain present, unbounded=True proves the bundle by k-induction."""
        assignments = assign_synapse_precisions(
            [np.array([[0.25, 0.75]])],
            target_error=0.05,
            min_bits=2,
            max_bits=8,
            min_length=8,
            max_length=16,
        )
        manifest = write_precision_formal_evidence_bundle(
            tmp_path, assignments, execute=True, unbounded=True
        )
        claim = manifest["formal_claim"]
        assert claim["symbiyosys_executed"] is True
        assert claim["formal_proof_passed"] is True
        assert claim["proof_mode"] == "prove"
        assert claim["proof_unbounded"] is True

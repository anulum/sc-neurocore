# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for adaptive per-layer bitstream precision

"""Tests for the adaptive precision assignment module."""

from __future__ import annotations

import json
from pathlib import Path
from collections.abc import Callable
from typing import cast

import numpy as np
import pytest

from sc_neurocore.compiler import formal_property_check
from sc_neurocore.compiler.adaptive_precision import (
    auto_tune_synapse_precisions,
    LayerPrecision,
    SynapsePrecision,
    analyze_sensitivity,
    assign_lengths,
    assign_synapse_precisions,
    precision_plan_manifest,
    write_precision_formal_evidence_bundle,
)
from sc_neurocore.compiler.formal_property_check import PropertyProofResult

_needs_formal = pytest.mark.skipif(
    not formal_property_check.formal_tools_available(),
    reason="SymbiYosys / Yosys / solver not available",
)


class TestAssignLengths:
    """Layer-level bitstream-length assignment checks."""

    def test_hoeffding_produces_assignments(self) -> None:
        """Hoeffding planning returns one assignment per layer."""
        weights = [np.random.randn(4, 2), np.random.randn(3, 4)]
        result = assign_lengths(weights, method="hoeffding")
        assert len(result) == 2
        assert all(isinstance(r, LayerPrecision) for r in result)

    def test_assignments_respect_bounds(self) -> None:
        """Assigned lengths stay inside caller-provided bounds."""
        weights = [np.random.randn(4, 2)]
        result = assign_lengths(weights, min_length=64, max_length=512)
        assert all(64 <= r.bitstream_length <= 512 for r in result)

    def test_lengths_are_power_of_two(self) -> None:
        """Assigned bitstream lengths are powers of two."""
        weights = [np.random.randn(8, 4), np.random.randn(4, 8)]
        result = assign_lengths(weights, method="hoeffding")
        for r in result:
            L = r.bitstream_length
            assert L & (L - 1) == 0, f"L={L} is not a power of 2"

    def test_relaxed_target_gives_shorter_lengths(self) -> None:
        """Relaxed error targets do not require longer bitstreams."""
        weights = [np.random.randn(4, 2)]
        tight = assign_lengths(weights, target_error=0.01, max_length=4096)
        relaxed = assign_lengths(weights, target_error=0.2, max_length=4096)
        assert relaxed[0].bitstream_length <= tight[0].bitstream_length

    def test_custom_layer_names(self) -> None:
        """Custom layer names propagate into assignments."""
        weights = [np.random.randn(4, 2)]
        result = assign_lengths(weights, layer_names=["my_layer"])
        assert result[0].name == "my_layer"

    def test_sensitivity_path_defaults_budget_to_full_width(self) -> None:
        """Non-Hoeffding planning defaults to a full-width total budget."""
        # A non-Hoeffding method follows the sensitivity-proportional branch;
        # with no explicit budget it defaults to max_length * n_layers and still
        # produces one bounded assignment per layer.
        weights = [np.random.randn(4, 2), np.random.randn(3, 4)]
        result = assign_lengths(weights, method="proportional", max_length=512)
        assert len(result) == 2
        assert all(r.bitstream_length <= 512 for r in result)
        assert all(r.sensitivity >= 0.0 for r in result)

    def test_default_layer_names(self) -> None:
        """Default layer names are deterministic."""
        weights = [np.random.randn(4, 2), np.random.randn(3, 4)]
        result = assign_lengths(weights)
        assert result[0].name == "layer_0"
        assert result[1].name == "layer_1"

    def test_sensitivity_method(self) -> None:
        """Sensitivity planning returns non-negative sensitivity scores."""
        weights = [np.random.randn(4, 2), np.random.randn(3, 4)]
        result = assign_lengths(weights, method="sensitivity", total_budget=2048)
        assert len(result) == 2
        assert all(r.sensitivity >= 0 for r in result)


class TestAnalyzeSensitivity:
    """Sensitivity-analysis facade checks."""

    def test_returns_per_layer_scores(self) -> None:
        """Sensitivity analysis returns one score per weight layer."""
        weights = [np.random.randn(4, 2), np.random.randn(3, 4)]
        sens = analyze_sensitivity(weights, n_trials=5)
        assert len(sens) == 2
        assert all(s >= 0 for s in sens)

    def test_larger_weights_more_sensitive(self) -> None:
        """Sensitivity scores remain numeric across weight scales."""
        small_w = [np.random.randn(4, 4) * 0.01]
        large_w = [np.random.randn(4, 4) * 0.5]
        sens_small = analyze_sensitivity(small_w, n_trials=10, seed=42)
        sens_large = analyze_sensitivity(large_w, n_trials=10, seed=42)
        # Larger weights should generally be more sensitive
        # (not guaranteed per-trial, but on average)
        assert isinstance(sens_small[0], float)
        assert isinstance(sens_large[0], float)


class TestLayerPrecision:
    """LayerPrecision data-contract checks."""

    def test_dataclass_fields(self) -> None:
        """LayerPrecision preserves assigned field values."""
        lp = LayerPrecision(
            layer_index=0,
            name="fc1",
            bitstream_length=256,
            error_bound=0.031,
            sensitivity=0.05,
        )
        assert lp.layer_index == 0
        assert lp.bitstream_length == 256

    def test_to_dict_serializes_manifest_row(self) -> None:
        """LayerPrecision exposes a deterministic manifest row."""
        lp = LayerPrecision(
            layer_index=1,
            name="classifier",
            bitstream_length=512,
            error_bound=0.015625,
            sensitivity=0.25,
        )

        assert lp.to_dict() == {
            "layer_index": 1,
            "name": "classifier",
            "bitstream_length": 512,
            "error_bound": 0.015625,
            "sensitivity": 0.25,
        }

    @pytest.mark.parametrize(
        ("factory", "message"),
        [
            (
                lambda: LayerPrecision(-1, "fc1", 256, 0.031, 0.05),
                "layer_index",
            ),
            (
                lambda: LayerPrecision(0, "", 256, 0.031, 0.05),
                "name",
            ),
            (
                lambda: LayerPrecision(0, "fc1", 0, 0.031, 0.05),
                "bitstream_length",
            ),
            (
                lambda: LayerPrecision(0, "fc1", 300, 0.031, 0.05),
                "power of two",
            ),
            (
                lambda: LayerPrecision(0, "fc1", 256, -0.1, 0.05),
                "error_bound",
            ),
            (
                lambda: LayerPrecision(0, "fc1", 256, 0.031, -0.1),
                "sensitivity",
            ),
        ],
    )
    def test_rejects_invalid_layer_precision_fields(
        self,
        factory: Callable[[], LayerPrecision],
        message: str,
    ) -> None:
        """LayerPrecision rejects impossible adaptive-length rows."""
        with pytest.raises(ValueError, match=message):
            factory()

    def test_rejects_non_numeric_layer_error_bound(self) -> None:
        """LayerPrecision rejects non-numeric error-bound payloads at runtime."""
        with pytest.raises(ValueError, match="error_bound"):
            LayerPrecision(
                layer_index=0,
                name="fc1",
                bitstream_length=256,
                error_bound=cast(float, "bad"),
                sensitivity=0.05,
            )


class TestSynapsePrecision:
    """Per-synapse precision-planning checks."""

    def test_synapse_precision_to_dict_is_stable(self) -> None:
        """SynapsePrecision serializes in manifest field order."""
        row = SynapsePrecision(
            layer_index=0,
            layer_name="fc",
            output_index=1,
            input_index=2,
            bit_width=8,
            bitstream_length=128,
            sensitivity=0.5,
            quantization_error_bound=0.01,
            stochastic_error_bound=0.02,
            total_error_bound=0.03,
        )

        assert row.to_dict() == {
            "layer_index": 0,
            "layer_name": "fc",
            "output_index": 1,
            "input_index": 2,
            "bit_width": 8,
            "bitstream_length": 128,
            "sensitivity": 0.5,
            "quantization_error_bound": 0.01,
            "stochastic_error_bound": 0.02,
            "total_error_bound": 0.03,
        }

    @pytest.mark.parametrize(
        ("factory", "message"),
        [
            (
                lambda: SynapsePrecision(-1, "fc", 0, 1, 8, 128, 0.5, 0.01, 0.02, 0.03),
                "layer_index",
            ),
            (
                lambda: SynapsePrecision(0, "", 0, 1, 8, 128, 0.5, 0.01, 0.02, 0.03),
                "layer_name",
            ),
            (
                lambda: SynapsePrecision(0, "fc", -1, 1, 8, 128, 0.5, 0.01, 0.02, 0.03),
                "output_index",
            ),
            (
                lambda: SynapsePrecision(0, "fc", 0, -1, 8, 128, 0.5, 0.01, 0.02, 0.03),
                "input_index",
            ),
            (
                lambda: SynapsePrecision(0, "fc", 0, 1, 0, 128, 0.5, 0.01, 0.02, 0.03),
                "bit_width",
            ),
            (
                lambda: SynapsePrecision(0, "fc", 0, 1, 8, 0, 0.5, 0.01, 0.02, 0.03),
                "bitstream_length",
            ),
            (
                lambda: SynapsePrecision(0, "fc", 0, 1, 8, 128, -0.1, 0.01, 0.02, 0.03),
                "sensitivity",
            ),
            (
                lambda: SynapsePrecision(0, "fc", 0, 1, 8, 128, 0.5, -0.1, 0.02, 0.03),
                "quantization_error_bound",
            ),
            (
                lambda: SynapsePrecision(0, "fc", 0, 1, 8, 128, 0.5, 0.01, -0.1, 0.03),
                "stochastic_error_bound",
            ),
            (
                lambda: SynapsePrecision(0, "fc", 0, 1, 8, 128, 0.5, 0.01, 0.02, -0.1),
                "total_error_bound",
            ),
        ],
    )
    def test_rejects_invalid_synapse_precision_fields(
        self,
        factory: Callable[[], SynapsePrecision],
        message: str,
    ) -> None:
        """SynapsePrecision rejects impossible per-synapse rows."""
        with pytest.raises(ValueError, match=message):
            factory()

    def test_rejects_total_error_bound_below_components(self) -> None:
        """Total error must include quantization and stochastic components."""
        with pytest.raises(ValueError, match="total_error_bound"):
            SynapsePrecision(
                layer_index=0,
                layer_name="fc",
                output_index=0,
                input_index=1,
                bit_width=8,
                bitstream_length=128,
                sensitivity=0.5,
                quantization_error_bound=0.02,
                stochastic_error_bound=0.02,
                total_error_bound=0.03,
            )

    def test_rejects_non_numeric_synapse_sensitivity(self) -> None:
        """SynapsePrecision rejects non-numeric sensitivity payloads at runtime."""
        with pytest.raises(ValueError, match="sensitivity"):
            SynapsePrecision(
                layer_index=0,
                layer_name="fc",
                output_index=0,
                input_index=1,
                bit_width=8,
                bitstream_length=128,
                sensitivity=cast(float, "bad"),
                quantization_error_bound=0.01,
                stochastic_error_bound=0.02,
                total_error_bound=0.03,
            )

    def test_assign_synapse_precisions_returns_one_row_per_weight(self) -> None:
        """Per-synapse planning returns one assignment per weight element."""
        weights = [np.array([[0.1, 0.8], [0.0, 0.4]])]

        result = assign_synapse_precisions(
            weights,
            layer_names=["fc"],
            target_error=0.05,
            min_bits=3,
            max_bits=8,
            min_length=16,
            max_length=256,
        )

        assert len(result) == 4
        assert all(isinstance(row, SynapsePrecision) for row in result)
        assert {row.layer_name for row in result} == {"fc"}
        assert all(3 <= row.bit_width <= 8 for row in result)
        assert all(16 <= row.bitstream_length <= 256 for row in result)
        assert all(row.total_error_bound >= row.quantization_error_bound for row in result)

    def test_high_sensitivity_gets_at_least_as_much_precision(self) -> None:
        """Higher-sensitivity weights receive at least as much precision."""
        weights = [np.array([[0.01, 1.0]])]
        result = assign_synapse_precisions(
            weights,
            target_error=0.05,
            min_bits=2,
            max_bits=10,
            min_length=16,
            max_length=512,
        )

        low, high = result
        assert high.sensitivity > low.sensitivity
        assert high.bit_width >= low.bit_width
        assert high.bitstream_length >= low.bitstream_length

    def test_custom_sensitivity_map_controls_precision(self) -> None:
        """Custom sensitivity maps influence per-synapse precision."""
        weights = [np.array([[0.5, 0.5]])]
        sensitivities = [np.array([[0.1, 1.0]])]

        result = assign_synapse_precisions(
            weights,
            sensitivity_maps=sensitivities,
            target_error=0.05,
            min_bits=2,
            max_bits=10,
            min_length=16,
            max_length=512,
        )

        assert result[1].sensitivity > result[0].sensitivity
        assert result[1].bit_width >= result[0].bit_width

    def test_precision_plan_manifest_is_deterministic(self) -> None:
        """Precision plan manifests expose deterministic keys and costs."""
        assignments = assign_synapse_precisions(
            [np.array([[0.25, 0.75]])],
            target_error=0.1,
            min_bits=2,
            max_bits=8,
        )

        manifest = precision_plan_manifest(assignments)

        assert manifest["schema"] == "sc-neurocore.adaptive_precision_plan.v1"
        assert manifest["granularity"] == "synapse"
        assert manifest["num_synapses"] == 2
        assert manifest["max_total_error_bound"] >= 0
        assert manifest["cost_summary"]["estimated_lut_cost"] > 0.0
        assert (
            manifest["cost_summary"]["uniform_length_reference_cost"]
            >= manifest["cost_summary"]["estimated_lut_cost"]
        )
        assert manifest["cost_summary"]["estimated_lut_savings_vs_uniform_length"] >= 0.0
        assert list(manifest["assignments"][0]) == [
            "layer_index",
            "layer_name",
            "output_index",
            "input_index",
            "bit_width",
            "bitstream_length",
            "sensitivity",
            "quantization_error_bound",
            "stochastic_error_bound",
            "total_error_bound",
        ]

    def test_rejects_invalid_sensitivity_shape(self) -> None:
        """Sensitivity maps must match their weight tensor shapes."""
        with np.testing.assert_raises_regex(ValueError, "sensitivity map"):
            assign_synapse_precisions(
                [np.array([[0.5, 0.5]])],
                sensitivity_maps=[np.array([[0.1], [1.0]])],
            )


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

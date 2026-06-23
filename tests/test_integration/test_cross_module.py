# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for cross-module integration (Actions #1-5 from gap analysis)."""

import numpy as np
import pytest


# ── Action #1: Rust backend wiring ────────────────────────────────────


class TestRustWiring:
    """Verify Rust backends are importable and callable from Python."""

    def test_optimizer_rust_import(self):
        try:
            from sc_neurocore_engine import py_opt_sa_search, py_opt_extract_pareto
        except ImportError:
            pytest.skip("Rust engine not built")
        assert callable(py_opt_sa_search)
        assert callable(py_opt_extract_pareto)

    def test_optimizer_sa_via_python_api(self):
        from sc_neurocore.optimizer.sc_optimizer import SCOptimizer, HardwareBudget, LayerProfile

        budget = HardwareBudget(max_luts=100_000, max_power_mw=1000.0)
        opt = SCOptimizer(budget)
        network = [
            LayerProfile(id="L0", mac_count=10),
            LayerProfile(id="L1", mac_count=20, is_critical_path=True),
        ]
        report = opt.optimize_annealing(network, max_iter=100)
        assert report is not None
        assert report.mean_accuracy > 0.5

    def test_evo_rust_functions(self):
        try:
            from sc_neurocore_engine import (
                py_evo_batch_mutate,
                py_evo_batch_fitness,
                py_evo_batch_crossover,
                py_evo_diversity,
                py_evo_novelty,
                py_evo_tournament,
            )
        except ImportError:
            pytest.skip("Rust engine not built")
        for _fn in (
            py_evo_batch_fitness,
            py_evo_batch_crossover,
            py_evo_diversity,
            py_evo_novelty,
            py_evo_tournament,
        ):
            assert callable(_fn)
        pop = [[0.0] * 10 for _ in range(20)]
        mutated = py_evo_batch_mutate(pop, 0.5, 0.1, 42)
        assert len(mutated) == 20
        assert any(w != 0.0 for g in mutated for w in g)

    def test_pareto_extraction_rust(self):
        try:
            from sc_neurocore_engine import py_opt_extract_pareto
        except ImportError:
            pytest.skip("Rust engine not built")
        result = py_opt_extract_pareto(
            [100, 200, 150],
            [1.0, 0.5, 0.8],
            [0.9, 0.95, 0.85],
        )
        assert "indices" in result
        assert len(result["luts"]) > 0


# ── Action #2: Shared core types ──────────────────────────────────────


class TestCoreTypes:
    """Verify shared type system works across modules."""

    def test_hardware_budget(self):
        from sc_neurocore.core.types import HardwareBudget

        b = HardwareBudget(max_luts=100_000)
        util = b.utilisation(luts=50_000)
        assert abs(util["luts"] - 0.5) < 1e-10

    def test_resource_report_meets_budget(self):
        from sc_neurocore.core.types import HardwareBudget, ResourceReport

        b = HardwareBudget(max_luts=100_000, max_power_mw=1000.0)
        r = ResourceReport(total_luts=50_000, total_power_mw=500.0)
        assert r.meets_budget(b)

    def test_resource_report_exceeds_budget(self):
        from sc_neurocore.core.types import HardwareBudget, ResourceReport

        b = HardwareBudget(max_luts=100_000, max_power_mw=1000.0)
        r = ResourceReport(total_luts=200_000, total_power_mw=500.0)
        assert not r.meets_budget(b)

    def test_layer_spec_estimation(self):
        from sc_neurocore.core.types import LayerSpec, DecorrelationStrategy

        ls = LayerSpec(
            layer_id="L0",
            neurons=64,
            bitstream_length=256,
            decorrelator=DecorrelationStrategy.SOBOL,
        )
        assert ls.estimate_luts() > 0
        assert ls.estimate_power_mw() > 0
        assert 0 < ls.estimate_accuracy() <= 1.0

    def test_estimate_network(self):
        from sc_neurocore.core.types import LayerSpec, estimate_network

        layers = [
            LayerSpec(layer_id="L0", neurons=32),
            LayerSpec(layer_id="L1", neurons=64),
        ]
        report = estimate_network(layers)
        assert report.total_luts > 0
        assert report.mean_accuracy > 0

    def test_layer_spec_deterministic(self):
        from sc_neurocore.core.types import LayerSpec, ComputeMode

        ls = LayerSpec(layer_id="L0", neurons=10, mode=ComputeMode.DETERMINISTIC)
        assert ls.estimate_accuracy() == 1.0

    def test_resource_report_exceeds_each_budget_dimension(self) -> None:
        # meets_budget returns on the first failing dimension, so each branch
        # needs a report that clears the earlier checks and breaches one.
        from sc_neurocore.core.types import HardwareBudget, ResourceReport

        power = ResourceReport(total_power_mw=2_000.0)
        assert not power.meets_budget(HardwareBudget(max_power_mw=1_000.0))

        latency = ResourceReport(total_latency_cycles=200)
        assert not latency.meets_budget(HardwareBudget(max_latency_cycles=100))

        ffs = ResourceReport(total_ffs=600_000)
        assert not ffs.meets_budget(HardwareBudget(max_ffs=500_000))

        dsp = ResourceReport(total_dsp=300)
        assert not dsp.meets_budget(HardwareBudget(max_dsp=256))

    def test_resource_report_summary_is_human_readable(self) -> None:
        from sc_neurocore.core.types import ResourceReport

        report = ResourceReport(
            total_luts=1_000,
            total_ffs=2_000,
            total_dsp=4,
            total_bram_kb=12.5,
            total_power_mw=3.25,
            total_latency_cycles=128,
            mean_accuracy=0.9876,
        )
        text = report.summary()
        assert "LUTs: 1000" in text
        assert "Accuracy: 0.9876" in text

    def test_layer_spec_deterministic_luts_and_power(self) -> None:
        # The deterministic mode takes the dedicated MAC-count cost paths in
        # estimate_luts and estimate_power_mw rather than the stochastic ones.
        from sc_neurocore.core.types import ComputeMode, LayerSpec

        ls = LayerSpec(layer_id="L0", neurons=8, mac_count=20, mode=ComputeMode.DETERMINISTIC)
        assert ls.estimate_luts() == 20 * 120
        assert ls.estimate_power_mw() == 20 * 0.5


# ── Action #3: Closed-loop adaptive controller ────────────────────────


class TestAdaptiveLoop:
    """Verify Runtime → Optimizer closed loop."""

    def test_controller_creation(self):
        from sc_neurocore.core.types import HardwareBudget, LayerSpec
        from sc_neurocore.control.adaptive_loop import AdaptiveController

        budget = HardwareBudget(max_luts=500_000, max_power_mw=5000.0)
        layers = [LayerSpec(layer_id="L0", neurons=64, mac_count=100)]
        ctrl = AdaptiveController(budget, layers)
        assert ctrl.current_report is None

    def test_no_drift_no_adaptation(self):
        from sc_neurocore.core.types import HardwareBudget, LayerSpec
        from sc_neurocore.control.adaptive_loop import AdaptiveController

        budget = HardwareBudget(max_luts=500_000, max_power_mw=5000.0)
        layers = [LayerSpec(layer_id="L0", neurons=10, mac_count=10)]
        ctrl = AdaptiveController(budget, layers)

        # Feed uncorrelated bitstreams (no drift)
        rng = np.random.default_rng(42)
        for _ in range(10):
            a = rng.integers(0, 2, size=256).astype(np.float64)
            b = rng.integers(0, 2, size=256).astype(np.float64)
            event = ctrl.step(a, b)
        assert len(ctrl.adaptation_log) == 0

    def test_drift_triggers_adaptation(self):
        from sc_neurocore.core.types import HardwareBudget, LayerSpec
        from sc_neurocore.control.adaptive_loop import AdaptiveController, AdaptiveLoopConfig

        budget = HardwareBudget(max_luts=500_000, max_power_mw=5000.0)
        layers = [LayerSpec(layer_id="L0", neurons=10, mac_count=10)]
        config = AdaptiveLoopConfig(
            drift_threshold=0.05,
            reoptimize_cooldown_s=0.0,
            sa_max_iter=50,
        )
        ctrl = AdaptiveController(budget, layers, config)

        # Use 50%-density pattern so SCC is properly computed (not degenerate)
        rng = np.random.default_rng(42)
        pattern = rng.integers(0, 2, size=256).astype(np.float64)
        for _ in range(100):
            ctrl.step(pattern, pattern)  # identical ⇒ SCC=1.0 ⇒ drift

        assert len(ctrl.adaptation_log) >= 1

    def test_summary(self):
        from sc_neurocore.core.types import HardwareBudget, LayerSpec
        from sc_neurocore.control.adaptive_loop import AdaptiveController

        budget = HardwareBudget(max_luts=500_000, max_power_mw=5000.0)
        layers = [LayerSpec(layer_id="L0", neurons=10, mac_count=10)]
        ctrl = AdaptiveController(budget, layers)
        s = ctrl.summary()
        assert "AdaptiveController" in s

    def test_cooldown_suppresses_back_to_back_reoptimisation(self):
        from sc_neurocore.core.types import HardwareBudget, LayerSpec
        from sc_neurocore.control.adaptive_loop import (
            AdaptiveController,
            AdaptiveLoopConfig,
        )

        budget = HardwareBudget(max_luts=500_000, max_power_mw=5000.0)
        layers = [LayerSpec(layer_id="L0", neurons=10, mac_count=10)]
        config = AdaptiveLoopConfig(
            drift_threshold=0.05,
            reoptimize_cooldown_s=100.0,
            sa_max_iter=50,
        )
        ctrl = AdaptiveController(budget, layers, config)
        rng = np.random.default_rng(42)
        pattern = rng.integers(0, 2, size=256).astype(np.float64)

        # Drive identical pairs until the first re-optimisation fires...
        first = None
        for _ in range(100):
            event = ctrl.step(pattern, pattern)
            if event is not None:
                first = event
                break
        assert first is not None
        # ...the very next step lands inside the 100 s cooldown window and is
        # suppressed, and the adaptation-rate property stays well defined.
        assert ctrl.step(pattern, pattern) is None
        assert 0.0 <= ctrl.adaptation_rate <= 1.0


# ── Action #4: Unified energy reporter ────────────────────────────────


class TestUnifiedEnergyReporter:
    """Verify Sustainability ↔ Profiling integration."""

    def test_basic_analysis(self):
        from sc_neurocore.energy_accounting.unified_reporter import UnifiedEnergyReporter
        from sc_neurocore.energy_accounting.sustainability_profiler import GridRegion

        reporter = UnifiedEnergyReporter(region=GridRegion.EU)
        report = reporter.analyze(
            layer_configs=[{"name": "L0", "power_mw": 10.0}],
            inference_time_s=0.001,
        )
        assert report.summary().startswith("Unified Energy Report")
        assert report.total_power_mw >= 10.0

    def test_asic_power_included(self):
        from sc_neurocore.energy_accounting.unified_reporter import UnifiedEnergyReporter

        reporter = UnifiedEnergyReporter(asic_power_mw=500.0)
        report = reporter.analyze(inference_time_s=0.001)
        assert report.asic_power_mw == 500.0
        assert report.total_power_mw >= 500.0

    def test_thermal_check(self):
        from sc_neurocore.energy_accounting.unified_reporter import UnifiedEnergyReporter

        reporter = UnifiedEnergyReporter()
        report = reporter.analyze(total_power_mw=100.0)
        assert report.junction_temp_c > 25.0
        assert report.thermal_safe


# ── Action #5: End-to-end export pipeline ─────────────────────────────


class TestExportPipeline:
    """Verify Model Zoo → ONNX → TVM → MLIR → Verilog pipeline."""

    def test_pipeline_creation(self):
        from sc_neurocore.export.pipeline import ExportPipeline

        p = ExportPipeline()
        assert p.registry is not None

    def test_pipeline_result_dataclass(self):
        from sc_neurocore.export.pipeline import PipelineResult, PipelineStageResult

        r = PipelineResult()
        r.stages.append(PipelineStageResult(stage="test", success=True, output="ok"))
        assert r.success
        assert "test" in r.summary()

    def test_pipeline_stage_failure(self):
        from sc_neurocore.export.pipeline import PipelineResult, PipelineStageResult

        r = PipelineResult()
        r.stages.append(PipelineStageResult(stage="fail", success=False, output="err"))
        assert not r.success

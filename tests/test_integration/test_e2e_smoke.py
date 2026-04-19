# SPDX-License-Identifier: AGPL-3.0-or-later
"""End-to-end smoke test: chains every subsystem in one flow.

Flow: Neuron Plugin → NAS Search → Optimizer → Runtime → Energy Report
"""

import numpy as np


class TestEndToEndPipeline:
    """Chains all subsystems in production order."""

    def test_full_flow_nas_to_energy(self):
        """NAS → Optimizer → Runtime → Energy → Export."""
        # Step 1: NAS search — find a Pareto-optimal architecture
        from sc_neurocore.nas.sc_nas_engine import run_nas

        report = run_nas(
            population_size=10,
            num_generations=5,
            seed=42,
        )
        assert len(report.pareto_front) > 0
        best = max(report.pareto_front, key=lambda c: c.accuracy)
        assert best.accuracy > 0

        # Step 2: Optimizer — optimise a network matching NAS result
        from sc_neurocore.optimizer.sc_optimizer import (
            SCOptimizer,
            HardwareBudget,
            LayerProfile,
        )

        budget = HardwareBudget(max_luts=500_000, max_power_mw=5000.0)
        opt = SCOptimizer(budget)
        network = [
            LayerProfile(id=f"L{i}", mac_count=layer.neurons) for i, layer in enumerate(best.layers)
        ]
        opt_report = opt.optimize_annealing(network, max_iter=100, seed=42)
        assert opt_report is not None
        assert opt_report.mean_accuracy > 0

        # Step 3: Shared types — estimate resources
        from sc_neurocore.core.types import LayerSpec, estimate_network

        layer_specs = [
            LayerSpec(
                layer_id=f"L{i}",
                neurons=layer.neurons,
                bitstream_length=layer.bitstream_length,
            )
            for i, layer in enumerate(best.layers)
        ]
        resources = estimate_network(layer_specs)
        assert resources.total_luts > 0
        assert resources.mean_accuracy > 0

        # Step 4: Energy report
        from sc_neurocore.energy_accounting.unified_reporter import UnifiedEnergyReporter

        reporter = UnifiedEnergyReporter()
        energy = reporter.analyze(total_power_mw=resources.total_luts * 0.01)
        assert energy.junction_temp_c > 25.0
        assert energy.thermal_safe

        # Step 5: Export pipeline — generate Verilog for best
        from sc_neurocore.nas.sc_nas_engine import NASVerilogEmitter

        verilog = NASVerilogEmitter.emit(best)
        assert "module" in verilog
        assert "endmodule" in verilog
        for i, layer in enumerate(best.layers):
            assert f"L{i}_NEURONS" in verilog

    def test_closed_loop_with_optimizer(self):
        """Verify adaptive loop triggers and produces valid config."""
        from sc_neurocore.core.types import HardwareBudget, LayerSpec
        from sc_neurocore.control.adaptive_loop import (
            AdaptiveController,
            AdaptiveLoopConfig,
        )

        budget = HardwareBudget(max_luts=500_000, max_power_mw=5000.0)
        layers = [
            LayerSpec(layer_id="L0", neurons=32, mac_count=32),
            LayerSpec(layer_id="L1", neurons=16, mac_count=16),
        ]
        config = AdaptiveLoopConfig(
            drift_threshold=0.05,
            reoptimize_cooldown_s=0.0,
            sa_max_iter=50,
        )
        ctrl = AdaptiveController(budget, layers, config)

        # Drive drift
        rng = np.random.default_rng(42)
        pattern = rng.integers(0, 2, size=256).astype(np.float64)
        for _ in range(100):
            ctrl.step(pattern, pattern)

        assert len(ctrl.adaptation_log) >= 1
        assert ctrl.current_config is not None
        assert ctrl.current_config.bitstream_length > 0

        # Verify summary works
        s = ctrl.summary()
        assert "adaptations" in s
        assert ctrl.current_report.mean_accuracy > 0

    def test_model_zoo_to_verilog(self):
        """Model Zoo neuron → NAS Verilog emission."""
        from sc_neurocore.nas.sc_nas_engine import (
            SCCandidate,
            LayerConfig,
            NeuronType,
            DecorrelationStrategy,
            NASVerilogEmitter,
        )

        candidate = SCCandidate(
            layers=[
                LayerConfig(
                    neurons=64,
                    neuron_type=NeuronType.LIF,
                    bitstream_length=256,
                    decorrelation=DecorrelationStrategy.SOBOL,
                ),
                LayerConfig(
                    neurons=32,
                    neuron_type=NeuronType.IZHIKEVICH,
                    bitstream_length=512,
                    decorrelation=DecorrelationStrategy.HALTON,
                ),
            ]
        )
        candidate.evaluate_resources()

        # Emit Verilog
        verilog = NASVerilogEmitter.emit(candidate, module_name="test_net")
        assert "module test_net" in verilog
        assert "sc_lif_neuron" in verilog
        assert "sc_izhikevich_neuron" in verilog
        assert "endmodule" in verilog

    def test_energy_carbon_chain(self):
        """Energy → Carbon → Thermal → Safety check."""
        from sc_neurocore.energy_accounting.unified_reporter import UnifiedEnergyReporter
        from sc_neurocore.energy_accounting.sustainability_profiler import GridRegion

        for region in [GridRegion.EU, GridRegion.US, GridRegion.CN]:
            reporter = UnifiedEnergyReporter(region=region, asic_power_mw=100.0)
            report = reporter.analyze(total_power_mw=500.0)
            assert report.carbon_g_co2 >= 0
            assert report.junction_temp_c > 25.0
            assert report.asic_power_mw == 100.0
            assert report.grid_region == region.value

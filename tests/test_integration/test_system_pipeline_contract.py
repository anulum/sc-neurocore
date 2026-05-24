# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for NAS-to-energy system pipeline contract

"""Named workflow contract: NAS search -> optimiser -> resources -> energy -> Verilog."""

from __future__ import annotations

import numpy as np


def test_nas_candidate_flows_through_optimizer_energy_and_verilog_export() -> None:
    from sc_neurocore.core.types import LayerSpec, estimate_network
    from sc_neurocore.energy_accounting.unified_reporter import UnifiedEnergyReporter
    from sc_neurocore.nas.sc_nas_engine import NASVerilogEmitter, run_nas
    from sc_neurocore.optimizer.sc_optimizer import HardwareBudget, LayerProfile, SCOptimizer

    report = run_nas(population_size=10, num_generations=5, seed=42)
    assert len(report.pareto_front) > 0
    best = max(report.pareto_front, key=lambda candidate: candidate.accuracy)
    assert best.accuracy > 0

    budget = HardwareBudget(max_luts=500_000, max_power_mw=5000.0)
    optimiser = SCOptimizer(budget)
    network = [
        LayerProfile(id=f"L{i}", mac_count=layer.neurons) for i, layer in enumerate(best.layers)
    ]
    opt_report = optimiser.optimize_annealing(network, max_iter=100, seed=42)
    assert opt_report is not None
    assert opt_report.mean_accuracy > 0

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

    energy = UnifiedEnergyReporter().analyze(total_power_mw=resources.total_luts * 0.01)
    assert energy.junction_temp_c > 25.0
    assert energy.thermal_safe

    verilog = NASVerilogEmitter.emit(best)
    assert "module" in verilog
    assert "endmodule" in verilog
    for i, _layer in enumerate(best.layers):
        assert f"L{i}_NEURONS" in verilog


def test_adaptive_controller_reoptimises_after_drift() -> None:
    from sc_neurocore.control.adaptive_loop import AdaptiveController, AdaptiveLoopConfig
    from sc_neurocore.core.types import HardwareBudget, LayerSpec

    budget = HardwareBudget(max_luts=500_000, max_power_mw=5000.0)
    layers = [
        LayerSpec(layer_id="L0", neurons=32, mac_count=32),
        LayerSpec(layer_id="L1", neurons=16, mac_count=16),
    ]
    controller = AdaptiveController(
        budget,
        layers,
        AdaptiveLoopConfig(drift_threshold=0.05, reoptimize_cooldown_s=0.0, sa_max_iter=50),
    )
    rng = np.random.default_rng(42)
    pattern = rng.integers(0, 2, size=256).astype(np.float64)

    for _ in range(100):
        controller.step(pattern, pattern)

    assert len(controller.adaptation_log) >= 1
    assert controller.current_config is not None
    assert controller.current_config.bitstream_length > 0
    assert "adaptations" in controller.summary()
    assert controller.current_report.mean_accuracy > 0


def test_nas_verilog_emitter_preserves_model_zoo_layer_types() -> None:
    from sc_neurocore.nas.sc_nas_engine import (
        DecorrelationStrategy,
        LayerConfig,
        NASVerilogEmitter,
        NeuronType,
        SCCandidate,
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

    verilog = NASVerilogEmitter.emit(candidate, module_name="test_net")

    assert "module test_net" in verilog
    assert "sc_lif_neuron" in verilog
    assert "sc_izhikevich_neuron" in verilog
    assert "endmodule" in verilog

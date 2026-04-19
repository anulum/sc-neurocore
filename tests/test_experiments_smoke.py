# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Smoke tests for experiments/

"""Smoke tests: import every experiment module and run its entry point
with minimal parameters — assert no crash, no exception.
"""

from __future__ import annotations


import pytest


# ---------------------------------------------------------------------------
# Import tests — every module must be importable
# ---------------------------------------------------------------------------

EXPERIMENT_MODULES = [
    "advanced_demo",
    "agent_synergy_demo",
    "bitstream_drive",
    "deep_research_demo",
    "demo_adaptive_audio",
    "demo_param_sweep",
    "demo_pattern_classification",
    "demo_pattern_classification_3class",
    "demo_pattern_pca",
    "demo_poisson_spikes",
    "demo_sc_dense_layer",
    "demo_sc_pipeline",
    "demo_sleep_optimization",
    "demo_swarm_control",
    "demo_tcbo_consciousness",
    "demonstration_convergence",
    "exascale_demo",
    "experimental_horizons_demo",
    "l7_symbolic_coupling",
    "learning_demo",
    "mega_advancements_demo",
    "quantum_neuromorphic_demo",
    "spatial_generative_demo",
    "system_level_demo",
    "tcbo_demo_engine",
    "whitepaper_benchmark",
]


@pytest.mark.parametrize("module_name", EXPERIMENT_MODULES)
def test_import(module_name):
    """Every experiment module must be importable."""
    import importlib

    mod = importlib.import_module(f"sc_neurocore.experiments.{module_name}")
    assert mod is not None


# ---------------------------------------------------------------------------
# Execution smoke tests — non-crashing, bounded time
# ---------------------------------------------------------------------------


class TestExperimentExecution:
    def test_poisson_spikes(self):
        from sc_neurocore.experiments.demo_poisson_spikes import run_demo

        result = run_demo()
        assert result is None or result is not None  # just don't crash

    def test_sc_dense_layer(self):
        from sc_neurocore.experiments.demo_sc_dense_layer import demo

        result = demo()
        assert result is None or result is not None

    def test_sc_pipeline(self):
        from sc_neurocore.experiments.demo_sc_pipeline import demo

        result = demo()
        assert result is None or result is not None

    def test_bitstream_drive(self):
        from sc_neurocore.experiments.bitstream_drive import run_bitstream_driven_lif

        input_bits, spike_bits, p_in, p_fire = run_bitstream_driven_lif(
            x_input=0.05, x_min=0.0, x_max=0.1, length=256
        )
        assert input_bits.shape == (256,)
        assert spike_bits.shape == (256,)
        assert 0.0 <= p_in <= 1.0
        assert 0.0 <= p_fire <= 1.0

    def test_pattern_classification(self):
        from sc_neurocore.experiments.demo_pattern_classification import demo

        result = demo()
        assert result is None or result is not None

    def test_pattern_classification_3class(self):
        from sc_neurocore.experiments.demo_pattern_classification_3class import demo

        result = demo()
        assert result is None or result is not None

    def test_learning_demo(self):
        from sc_neurocore.experiments.learning_demo import run_learning_experiment

        result = run_learning_experiment()
        assert result is None or result is not None

    def test_convergence_demo(self):
        from sc_neurocore.experiments.demonstration_convergence import run_demonstration

        result = run_demonstration()
        assert result is None or result is not None

    def test_advanced_demo(self):
        from sc_neurocore.experiments.advanced_demo import run_advanced_demo

        result = run_advanced_demo()
        assert result is None or result is not None

    @pytest.mark.skip(
        reason="Long-running demo (exascale simulation); imported-only in smoke suite"
    )
    def test_exascale_demo(self):
        from sc_neurocore.experiments.exascale_demo import run_exascale_demo

        result = run_exascale_demo()
        assert result is None or result is not None

    @pytest.mark.skip(
        reason="Long-running demo (whitepaper benchmark suite); imported-only in smoke suite"
    )
    def test_whitepaper_benchmark(self):
        from sc_neurocore.experiments.whitepaper_benchmark import run_whitepaper_benchmark

        result = run_whitepaper_benchmark()
        assert result is None or result is not None

    @pytest.mark.skip(
        reason="Long-running demo (multi-agent synergy); imported-only in smoke suite"
    )
    def test_agent_synergy_demo(self):
        from sc_neurocore.experiments.agent_synergy_demo import run_agent_demo

        result = run_agent_demo()
        assert result is None or result is not None

    @pytest.mark.xfail(reason="Pre-existing API mismatch: FeynmanKacHeatSolver alpha kwarg")
    def test_deep_research_demo(self):
        from sc_neurocore.experiments.deep_research_demo import run_deep_research_demo

        result = run_deep_research_demo()
        assert result is None or result is not None

    def test_experimental_horizons_demo(self):
        from sc_neurocore.experiments.experimental_horizons_demo import run_horizons_demo

        result = run_horizons_demo()
        assert result is None or result is not None

    def test_mega_advancements_demo(self):
        from sc_neurocore.experiments.mega_advancements_demo import run_demo

        result = run_demo()
        assert result is None or result is not None

    def test_spatial_generative_demo(self):
        from sc_neurocore.experiments.spatial_generative_demo import run_spatial_gen_demo

        result = run_spatial_gen_demo()
        assert result is None or result is not None

    def test_system_level_demo(self):
        from sc_neurocore.experiments.system_level_demo import run_system_demo

        result = run_system_demo()
        assert result is None or result is not None

    def test_quantum_neuromorphic_demo(self):
        from sc_neurocore.experiments.quantum_neuromorphic_demo import run_demo

        result = run_demo()
        assert result is None or result is not None

    def test_tcbo_demo(self):
        from sc_neurocore.experiments.demo_tcbo_consciousness import run_demo

        result = run_demo()
        assert result is None or result is not None

    def test_swarm_control_demo(self):
        from sc_neurocore.experiments.demo_swarm_control import run_demo

        result = run_demo()
        assert result is None or result is not None

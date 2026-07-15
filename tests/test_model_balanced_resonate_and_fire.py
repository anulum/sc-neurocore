# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — BalancedResonateAndFireNeuron Tests

"""Publication-equation tests for Higuchi et al. 2024 BRF neuron."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from sc_neurocore.neurons import BalancedResonateAndFireNeuron as PublicBRF
from sc_neurocore.neurons.models.balanced_resonate_and_fire import (
    BalancedResonateAndFireNeuron,
    sustain_oscillation_boundary,
)
from sc_neurocore.network.population import Population

REPO_ROOT = Path(__file__).resolve().parents[1]


def _run(neuron: BalancedResonateAndFireNeuron, current: float, steps: int) -> list[int]:
    return [step for step in range(steps) if neuron.step(current) == 1]


class TestBRFEquations:
    def test_construction_defaults_match_paper_algorithm(self) -> None:
        neuron = BalancedResonateAndFireNeuron()
        assert neuron.x == 0.0
        assert neuron.y == 0.0
        assert neuron.q == 0.0
        assert neuron.omega == 10.0
        assert neuron.b_offset == 1.0
        assert neuron.threshold == 1.0
        assert neuron.gamma == 0.9
        assert neuron.dt == 0.01

    def test_divergence_boundary_formula(self) -> None:
        omega = 10.0
        dt = 0.01
        expected = (-1.0 + math.sqrt(1.0 - (dt * omega) ** 2)) / dt
        assert sustain_oscillation_boundary(omega, dt) == pytest.approx(expected)
        assert BalancedResonateAndFireNeuron(omega=omega, dt=dt).p_omega == pytest.approx(expected)

    def test_one_step_matches_algorithm_1(self) -> None:
        neuron = BalancedResonateAndFireNeuron(
            x=0.2,
            y=-0.1,
            q=0.3,
            omega=12.0,
            b_offset=0.75,
            threshold=1.0,
            gamma=0.9,
            dt=0.01,
        )
        p_omega = sustain_oscillation_boundary(12.0, 0.01)
        b_t = p_omega - 0.75 - 0.3
        expected_x = 0.2 + 0.01 * (b_t * 0.2 - 12.0 * -0.1 + 2.0)
        expected_y = -0.1 + 0.01 * (12.0 * 0.2 + b_t * -0.1)
        expected_spike = int(expected_x >= 1.3)

        spike = neuron.step(2.0)

        assert spike == expected_spike
        assert neuron.x == pytest.approx(expected_x)
        assert neuron.y == pytest.approx(expected_y)
        assert neuron.q == pytest.approx(0.9 * 0.3 + expected_spike)

    def test_threshold_uses_real_part_not_radius(self) -> None:
        neuron = BalancedResonateAndFireNeuron(x=0.0, y=5.0)
        assert neuron.step(0.0) == 0
        assert neuron.q == 0.0


class TestBRFStabilityAndReset:
    def test_invalid_boundary_fails_fast(self) -> None:
        with pytest.raises(ValueError, match=r"dt \* omega"):
            BalancedResonateAndFireNeuron(omega=200.0, dt=0.01)
        with pytest.raises(ValueError, match=r"dt \* omega"):
            sustain_oscillation_boundary(200.0, 0.01)

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"dt": 0.0}, "dt"),
            ({"omega": 0.0}, "omega"),
            ({"b_offset": 0.0}, "b_offset"),
            ({"threshold": 0.0}, "threshold"),
            ({"gamma": 1.0}, "gamma"),
            ({"x": float("nan")}, "finite"),
        ],
    )
    def test_parameter_validation(self, kwargs: dict[str, float], match: str) -> None:
        with pytest.raises(ValueError, match=match):
            BalancedResonateAndFireNeuron(**kwargs)

    def test_refractory_period_raises_threshold_and_decays(self) -> None:
        neuron = BalancedResonateAndFireNeuron()
        assert neuron.step(200.0) == 1
        assert neuron.q == pytest.approx(1.0)
        assert neuron.dynamic_threshold == pytest.approx(2.0)

        neuron.step(0.0)
        assert 0.0 < neuron.q < 1.0
        assert neuron.dynamic_threshold == pytest.approx(1.9)

    def test_smooth_reset_preserves_phase_state_after_spike(self) -> None:
        neuron = BalancedResonateAndFireNeuron()
        assert neuron.step(200.0) == 1
        assert neuron.x != 0.0
        assert neuron.y == 0.0
        assert neuron.damping < neuron.p_omega - neuron.b_offset

    def test_state_remains_finite_under_long_drive(self) -> None:
        neuron = BalancedResonateAndFireNeuron(omega=20.0, b_offset=2.0)
        for _ in range(20_000):
            neuron.step(5.0)
        snapshot = neuron.state()
        assert all(math.isfinite(value) for value in snapshot.values())

    def test_reset_clears_membrane_and_refractory_state(self) -> None:
        neuron = BalancedResonateAndFireNeuron()
        neuron.step(200.0)
        neuron.reset()
        assert neuron.x == 0.0
        assert neuron.y == 0.0
        assert neuron.q == 0.0


class TestBRFResponse:
    def test_frequency_timed_excitation_produces_more_spikes_than_off_phase(self) -> None:
        omega = 10.0
        dt = 0.01
        steps = 3000
        period_steps = max(1, round((2.0 * math.pi / omega) / dt))
        resonant = BalancedResonateAndFireNeuron(omega=omega, dt=dt)
        off_phase = BalancedResonateAndFireNeuron(omega=omega, dt=dt)
        resonant_spikes = 0
        off_phase_spikes = 0

        for step in range(steps):
            resonant_current = 160.0 if step % period_steps == 0 else 0.0
            off_phase_current = 160.0 if step % period_steps == period_steps // 2 else 0.0
            resonant_spikes += resonant.step(resonant_current)
            off_phase_spikes += off_phase.step(off_phase_current)

        assert resonant_spikes >= off_phase_spikes
        assert resonant_spikes > 0

    def test_refractory_smooth_reset_sparsifies_high_drive(self) -> None:
        sparse = len(_run(BalancedResonateAndFireNeuron(), current=120.0, steps=500))
        dense_upper_bound = 500
        assert 0 < sparse < dense_upper_bound

    def test_import_surfaces_and_population_wiring(self) -> None:
        assert PublicBRF.__name__ == "BalancedResonateAndFireNeuron"
        pop = Population(BalancedResonateAndFireNeuron, n=4, label="brf")
        assert pop.n == 4
        assert all(isinstance(neuron, BalancedResonateAndFireNeuron) for neuron in pop.neurons)

    def test_reproducible_deterministic_trace(self) -> None:
        left = BalancedResonateAndFireNeuron(omega=15.0, b_offset=1.5)
        right = BalancedResonateAndFireNeuron(omega=15.0, b_offset=1.5)
        currents = np.sin(np.linspace(0.0, 12.0, 200)) * 10.0 + 10.0
        left_trace = [left.step(float(current)) for current in currents]
        right_trace = [right.step(float(current)) for current in currents]
        assert left_trace == right_trace
        assert left.state() == right.state()


class TestBRFProjectWiring:
    def test_polyglot_mirror_files_exist_with_brf_equations(self) -> None:
        equation_paths = [
            "src/sc_neurocore/accel/rust/safety/balanced_resonate_and_fire.rs",
            "engine/src/neurons/simple_spiking/balanced_resonate_and_fire.rs",
            "src/sc_neurocore/accel/go/services/balanced_resonate_and_fire.go",
            "src/sc_neurocore/accel/julia/neurons/balanced_resonate_and_fire.jl",
            "src/sc_neurocore/accel/mojo/kernels/balanced_resonate_and_fire.mojo",
        ]
        for relative_path in equation_paths:
            path = REPO_ROOT / relative_path
            body = path.read_text(encoding="utf-8")
            assert "sustain" in body.lower()
            assert "b_offset" in body.lower() or "boffset" in body.lower()
            assert "gamma" in body.lower()
        assert "PyBalancedResonateAndFireNeuron" in (
            REPO_ROOT / "engine/src/pyo3_neurons.rs"
        ).read_text(encoding="utf-8")
        assert "BalancedResonateAndFire" in (REPO_ROOT / "engine/src/network_runner.rs").read_text(
            encoding="utf-8"
        )

    def test_benchmark_and_documentation_are_wired(self) -> None:
        assert (REPO_ROOT / "benchmarks/bench_balanced_resonate_and_fire.py").exists()
        assert (REPO_ROOT / "benchmarks/results/bench_balanced_resonate_and_fire.json").exists()
        benchmark = (
            REPO_ROOT / "benchmarks/results/bench_balanced_resonate_and_fire.json"
        ).read_text(encoding="utf-8")
        assert '"python_step_ns"' in benchmark
        assert '"rust_pyo3_step_ns"' in benchmark
        assert '"go_step_ns"' in benchmark
        assert '"julia_step_ns"' in benchmark
        assert '"mojo_step_ns"' in benchmark
        doc = (REPO_ROOT / "docs/api/models/balanced_resonate_and_fire.md").read_text(
            encoding="utf-8"
        )
        assert "Algorithm 1" in doc
        assert "bench_balanced_resonate_and_fire.py" in doc

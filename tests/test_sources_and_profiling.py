"""
Tests for untested source and profiling modules:
  - sources/quantum_entropy.py (QuantumEntropySource)
  - sources/bitstream_current_source.py (BitstreamCurrentSource)
  - profiling/energy.py (EnergyMetrics, track_energy)
"""

import pytest
import numpy as np

from sc_neurocore.sources.quantum_entropy import QuantumEntropySource
from sc_neurocore.sources.bitstream_current_source import BitstreamCurrentSource
from sc_neurocore.profiling.energy import EnergyMetrics, profiler, track_energy


# ---------------------------------------------------------------------------
# QuantumEntropySource
# ---------------------------------------------------------------------------

class TestQuantumEntropySource:
    def test_construction_default(self):
        qes = QuantumEntropySource(n_qubits=1, seed=42)
        assert qes.state.shape == (2,)
        assert qes.state[0] == 1.0 + 0j

    def test_construction_multi_qubit(self):
        qes = QuantumEntropySource(n_qubits=3, seed=0)
        assert qes.state.shape == (8,)  # 2^3

    def test_sample_normal_returns_float(self):
        qes = QuantumEntropySource(n_qubits=1, seed=0)
        val = qes.sample_normal()
        assert isinstance(val, float)

    def test_sample_returns_float(self):
        qes = QuantumEntropySource(n_qubits=1, seed=0)
        assert isinstance(qes.sample(), float)

    def test_hadamard_normalizes_state(self):
        """After Hadamard, state should remain normalised (sum |a|^2 = 1)."""
        qes = QuantumEntropySource(n_qubits=2, seed=42)
        qes._hadamard()
        norm = np.sum(np.abs(qes.state) ** 2)
        assert norm == pytest.approx(1.0, abs=1e-10)

    def test_samples_vary(self):
        """Repeated samples should not all be identical."""
        qes = QuantumEntropySource(n_qubits=1, seed=42)
        samples = [qes.sample_normal() for _ in range(50)]
        assert len(set(samples)) > 1

    def test_mean_std_scaling(self):
        """With many samples, mean and spread should roughly match request."""
        qes = QuantumEntropySource(n_qubits=3, seed=0)
        samples = np.array([qes.sample_normal(mean=5.0, std=2.0) for _ in range(2000)])
        # Not Gaussian, but mean should be near 5 and spread should be > 0
        assert abs(samples.mean() - 5.0) < 2.0
        assert samples.std() > 0.1

    def test_reproducible_with_seed(self):
        qes1 = QuantumEntropySource(n_qubits=1, seed=99)
        qes2 = QuantumEntropySource(n_qubits=1, seed=99)
        s1 = [qes1.sample() for _ in range(10)]
        s2 = [qes2.sample() for _ in range(10)]
        assert s1 == s2


# ---------------------------------------------------------------------------
# BitstreamCurrentSource
# ---------------------------------------------------------------------------

class TestBitstreamCurrentSource:
    def test_construction(self):
        src = BitstreamCurrentSource(
            x_inputs=[0.5, 0.5],
            x_min=0.0, x_max=1.0,
            weight_values=[0.5, 0.5],
            w_min=0.0, w_max=1.0,
            length=256, seed=42,
        )
        assert src.n_inputs == 2
        assert src.pre_matrix.shape == (2, 256)
        assert src.post_matrix.shape == (2, 256)

    def test_input_weight_mismatch_raises(self):
        with pytest.raises(ValueError):
            BitstreamCurrentSource(
                x_inputs=[0.5, 0.5],
                x_min=0.0, x_max=1.0,
                weight_values=[0.5],
                w_min=0.0, w_max=1.0,
            )

    def test_step_returns_float_in_range(self):
        src = BitstreamCurrentSource(
            x_inputs=[0.8],
            x_min=0.0, x_max=1.0,
            weight_values=[0.6],
            w_min=0.0, w_max=1.0,
            length=64, y_min=0.0, y_max=1.0, seed=42,
        )
        for _ in range(64):
            I_t = src.step()
            assert 0.0 <= I_t <= 1.0

    def test_step_clamps_past_length(self):
        """After length steps, it should clamp at the last index."""
        src = BitstreamCurrentSource(
            x_inputs=[0.5],
            x_min=0.0, x_max=1.0,
            weight_values=[0.5],
            w_min=0.0, w_max=1.0,
            length=8, seed=42,
        )
        for _ in range(20):
            I_t = src.step()
        assert isinstance(I_t, float)

    def test_reset(self):
        src = BitstreamCurrentSource(
            x_inputs=[0.5],
            x_min=0.0, x_max=1.0,
            weight_values=[0.5],
            w_min=0.0, w_max=1.0,
            length=16, seed=42,
        )
        first_vals = [src.step() for _ in range(5)]
        src.reset()
        second_vals = [src.step() for _ in range(5)]
        assert first_vals == second_vals

    def test_full_current_estimate(self):
        src = BitstreamCurrentSource(
            x_inputs=[0.8, 0.6],
            x_min=0.0, x_max=1.0,
            weight_values=[0.5, 0.5],
            w_min=0.0, w_max=1.0,
            length=1024, y_min=0.0, y_max=0.1, seed=42,
        )
        est = src.full_current_estimate()
        assert isinstance(est, float)
        assert 0.0 <= est <= 0.1

    def test_high_weight_high_input_gives_more_current(self):
        """Higher weights and inputs should produce more current."""
        src_low = BitstreamCurrentSource(
            x_inputs=[0.2], x_min=0.0, x_max=1.0,
            weight_values=[0.2], w_min=0.0, w_max=1.0,
            length=1024, y_min=0.0, y_max=1.0, seed=42,
        )
        src_high = BitstreamCurrentSource(
            x_inputs=[0.9], x_min=0.0, x_max=1.0,
            weight_values=[0.9], w_min=0.0, w_max=1.0,
            length=1024, y_min=0.0, y_max=1.0, seed=42,
        )
        assert src_high.full_current_estimate() > src_low.full_current_estimate()


# ---------------------------------------------------------------------------
# profiling/energy.py
# ---------------------------------------------------------------------------

class TestEnergyMetrics:
    def test_defaults(self):
        em = EnergyMetrics()
        assert em.total_ops_and == 0
        assert em.total_ops_xor == 0
        assert em.total_bits_mem == 0

    def test_estimate_energy_zero(self):
        em = EnergyMetrics()
        assert em.estimate_energy() == 0.0

    def test_estimate_energy_with_ops(self):
        em = EnergyMetrics()
        em.total_ops_and = 1_000_000
        energy = em.estimate_energy()
        expected = 1_000_000 * 0.1e-15
        assert energy == pytest.approx(expected)

    def test_estimate_energy_combined(self):
        em = EnergyMetrics()
        em.total_ops_and = 100
        em.total_ops_xor = 200
        em.total_bits_mem = 300
        energy = em.estimate_energy()
        expected = 100 * 0.1e-15 + 200 * 0.15e-15 + 300 * 5.0e-15
        assert energy == pytest.approx(expected)

    def test_reset(self):
        em = EnergyMetrics()
        em.total_ops_and = 999
        em.total_ops_xor = 888
        em.total_bits_mem = 777
        em.reset()
        assert em.total_ops_and == 0
        assert em.total_ops_xor == 0
        assert em.total_bits_mem == 0

    def test_co2_emission(self):
        em = EnergyMetrics()
        em.total_ops_and = 1_000_000_000  # 1 billion AND ops
        co2 = em.co2_emission_g()
        assert co2 > 0
        assert isinstance(co2, float)

    def test_co2_custom_intensity(self):
        em = EnergyMetrics()
        em.total_ops_and = 1_000_000
        co2_default = em.co2_emission_g(carbon_intensity_g_per_kwh=475)
        co2_green = em.co2_emission_g(carbon_intensity_g_per_kwh=50)
        assert co2_green < co2_default


class TestTrackEnergyDecorator:
    def test_decorator_accumulates_ops(self):
        """track_energy should add AND ops for a layer-like object."""
        profiler.reset()

        class MockLayer:
            n_neurons = 4
            n_inputs = 3
            length = 100

            @track_energy
            def forward(self):
                return "done"

        layer = MockLayer()
        layer.forward()
        # ops = n_inputs * n_neurons * length = 3 * 4 * 100 = 1200
        assert profiler.total_ops_and == 1200
        # mem = (n_neurons * n_inputs * length) + (n_inputs * length) = 1200 + 300
        assert profiler.total_bits_mem == 1500

    def test_decorator_returns_original_result(self):
        profiler.reset()

        class MockLayer:
            n_neurons = 2
            n_inputs = 2
            length = 10

            @track_energy
            def forward(self):
                return 42

        layer = MockLayer()
        assert layer.forward() == 42

    def test_decorator_no_layer_attributes(self):
        """Decorator should still work if the object lacks layer attributes."""
        profiler.reset()

        @track_energy
        def simple_func():
            return "ok"

        assert simple_func() == "ok"
        assert profiler.total_ops_and == 0  # nothing accumulated

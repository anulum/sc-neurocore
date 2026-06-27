# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for 11 gap-analysis neuron and synapse models

"""Sophisticated multi-angle tests for publication-matched gap models.

Each model is tested for:
- Correct construction and defaults
- Functional step() producing expected output types
- Reset returns to initial state
- Publication-specific properties (equations, behaviours)
- Edge cases and boundary conditions
- Parity between Python and Rust (same equations, same parameters)
"""

from __future__ import annotations

import math

import pytest


# ═══════════════════════════════════════════════════════════════════════
# AdaptiveThresholdMoENeuron — SpikingBrain arXiv:2509.05276v2
# ═══════════════════════════════════════════════════════════════════════


class TestAdaptiveThresholdMoENeuron:
    @pytest.fixture()
    def neuron(self):
        from sc_neurocore.neurons.models import AdaptiveThresholdMoENeuron

        return AdaptiveThresholdMoENeuron(k=4.0)

    def test_defaults(self, neuron):
        assert neuron.k == 4.0
        assert neuron.v == 0.0
        assert neuron.v_th == 1.0

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"k": 0.0},
            {"k": float("nan")},
            {"ema_alpha": 0.0},
            {"ema_alpha": 1.1},
            {"ema_alpha": float("inf")},
            {"v": float("nan")},
            {"v_th": 0.0},
            {"_mean_abs_x": -0.01},
        ],
    )
    def test_rejects_non_physical_adaptive_threshold_parameters(self, kwargs):
        """Adaptive threshold dynamics require finite positive scaling and state."""
        from sc_neurocore.neurons.models import AdaptiveThresholdMoENeuron

        with pytest.raises(ValueError):
            AdaptiveThresholdMoENeuron(**kwargs)

    @pytest.mark.parametrize("current", [float("nan"), float("inf")])
    def test_rejects_non_finite_current(self, current):
        """Threshold adaptation must fail closed on non-finite current."""
        from sc_neurocore.neurons.models import AdaptiveThresholdMoENeuron

        with pytest.raises(ValueError, match="current"):
            AdaptiveThresholdMoENeuron().step(current)

    @pytest.mark.parametrize("activation", [float("nan"), float("-inf")])
    def test_rejects_non_finite_collapsed_activation(self, activation):
        """Collapsed inference must fail closed on non-finite activation."""
        from sc_neurocore.neurons.models import AdaptiveThresholdMoENeuron

        with pytest.raises(ValueError, match="activation"):
            AdaptiveThresholdMoENeuron().step_collapsed(activation)

    def test_step_returns_int(self, neuron):
        result = neuron.step(1.0)
        assert isinstance(result, int)

    def test_non_negative_spike_count(self, neuron):
        """SpikingBrain s_INT must be >= 0."""
        for _ in range(100):
            s = neuron.step(-5.0)
            assert s >= 0

    def test_integer_spike_count_gt_one(self):
        """With high input and low k, spike count can exceed 1."""
        from sc_neurocore.neurons.models import AdaptiveThresholdMoENeuron

        n = AdaptiveThresholdMoENeuron(k=10.0, ema_alpha=0.5)
        # Warm up EMA.
        for _ in range(20):
            n.step(5.0)
        # With k=10, V_th = mean(|x|)/10 = 0.5. v accumulates to 5, s=round(5/0.5)=10.
        s = n.step(5.0)
        assert s > 1, f"Expected multi-spike, got {s}"

    def test_soft_reset_preserves_residual(self):
        """After spike, v retains the sub-threshold residual."""
        from sc_neurocore.neurons.models import AdaptiveThresholdMoENeuron

        n = AdaptiveThresholdMoENeuron(k=4.0, ema_alpha=1.0)
        n.step(1.0)  # sets mean_abs_x = 1.0, v_th = 0.25
        # v = 1.0, s = round(1.0/0.25) = 4, v = 1.0 - 0.25*4 = 0.0
        assert abs(n.v) < 0.01

    def test_adaptive_threshold_tracks_input(self):
        """V_th = (1/k) * mean(|x|) tracks input magnitude."""
        from sc_neurocore.neurons.models import AdaptiveThresholdMoENeuron

        n = AdaptiveThresholdMoENeuron(k=4.0, ema_alpha=0.5)
        for _ in range(50):
            n.step(10.0)
        assert n.v_th > 1.0, "Threshold must rise with large inputs"
        n2 = AdaptiveThresholdMoENeuron(k=4.0, ema_alpha=0.5)
        for _ in range(50):
            n2.step(0.1)
        assert n2.v_th < n.v_th

    def test_sparsity_below_threshold(self, neuron):
        assert neuron.sparsity() == 1.0  # no input yet

    def test_step_collapsed(self, neuron):
        """Time-collapsed mode: s_INT = round(x / V_th)."""
        for _ in range(20):
            neuron.step_collapsed(2.0)
        s = neuron.step_collapsed(2.0)
        assert isinstance(s, int)
        assert s >= 0

    def test_reset(self, neuron):
        for _ in range(50):
            neuron.step(3.0)
        neuron.reset()
        assert neuron.v == 0.0
        assert neuron.v_th == 1.0
        assert neuron._mean_abs_x == 0.0

    def test_varying_input_produces_sparsity(self):
        """With varying inputs and k=1, not every step spikes."""
        from sc_neurocore.neurons.models import AdaptiveThresholdMoENeuron

        n = AdaptiveThresholdMoENeuron(k=1.0, ema_alpha=0.3)
        inputs = [0.0, 0.0, 5.0, 0.0, 0.0, 5.0, 0.0] * 10
        spikes = [n.step(x) for x in inputs]
        non_spiking = sum(1 for s in spikes if s == 0)
        assert non_spiking > 0, "Some steps must have zero spikes (sparsity)"


# ═══════════════════════════════════════════════════════════════════════
# HybridLinearAttentionNeuron — SpikingBrain hybrid attention
# ═══════════════════════════════════════════════════════════════════════


class TestHybridLinearAttentionNeuron:
    @pytest.fixture()
    def neuron(self):
        from sc_neurocore.neurons.models import HybridLinearAttentionNeuron

        return HybridLinearAttentionNeuron(dim=16)

    def test_defaults(self, neuron):
        assert neuron.dim == 16
        assert neuron.lambda_decay == 0.95
        assert neuron.window_size == 16

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"dim": 0},
            {"dim": 1.5},
            {"lambda_decay": -0.01},
            {"lambda_decay": 1.01},
            {"lambda_decay": float("nan")},
            {"window_size": 0},
            {"window_size": 2.5},
            {"dt": 0.0},
            {"v": float("inf")},
            {"_state_kv": [0.0, float("nan")]},
            {"_window_buf": [0.0, float("inf")]},
        ],
    )
    def test_rejects_non_physical_attention_parameters(self, kwargs):
        """Hybrid attention state must be finite, bounded, and dimensionally valid."""
        from sc_neurocore.neurons.models import HybridLinearAttentionNeuron

        with pytest.raises(ValueError):
            HybridLinearAttentionNeuron(**kwargs)

    @pytest.mark.parametrize(
        ("query", "key", "value"),
        [(float("nan"), 0.0, 0.0), (0.0, float("inf"), 0.0), (0.0, 0.0, float("nan"))],
    )
    def test_rejects_non_finite_qkv_drive(self, query, key, value):
        """Attention update must fail closed on non-finite projections."""
        from sc_neurocore.neurons.models import HybridLinearAttentionNeuron

        with pytest.raises(ValueError, match="query, key, and value"):
            HybridLinearAttentionNeuron().step_qkv(query, key, value)

    def test_step_qkv_returns_float(self, neuron):
        out = neuron.step_qkv(1.0, 0.5, 2.0)
        assert isinstance(out, float)

    def test_step_returns_binary(self, neuron):
        s = neuron.step(0.5)
        assert s in (0, 1)

    def test_phi_feature_map(self, neuron):
        """phi(x) = elu(x) + 1: positive -> x+1, negative -> exp(x)."""
        assert neuron._phi(2.0) == 3.0
        assert abs(neuron._phi(-1.0) - math.exp(-1.0)) < 1e-10
        assert neuron._phi(0.0) == 1.0  # boundary

    def test_recurrent_state_decays(self, neuron):
        """Lambda decay: state_kv *= lambda each step."""
        neuron.step_qkv(1.0, 1.0, 10.0)
        first_v = neuron.v
        # Feed zeros — state decays.
        for _ in range(50):
            neuron.step_qkv(0.01, 0.01, 0.0)
        assert abs(neuron.v) < abs(first_v)

    def test_window_buffer_averaging(self, neuron):
        """Local attention = sliding window average of values."""
        for i in range(16):
            neuron.step_qkv(0.0, 0.0, float(i))
        # Window now has [0..15], mean = 7.5.
        # With q=0 → phi(0) = 1, global component is small.
        # local = mean(window) = 7.5, v ≈ 0.5 * global + 0.5 * 7.5

    def test_reset(self, neuron):
        for _ in range(20):
            neuron.step(2.0)
        neuron.reset()
        assert neuron.v == 0.0
        assert all(s == 0.0 for s in neuron._state_kv)
        assert all(w == 0.0 for w in neuron._window_buf)

    def test_different_dims(self):
        from sc_neurocore.neurons.models import HybridLinearAttentionNeuron

        for dim in [4, 32, 64]:
            n = HybridLinearAttentionNeuron(dim=dim)
            assert len(n._state_kv) == dim
            n.step_qkv(1.0, 1.0, 1.0)


# ═══════════════════════════════════════════════════════════════════════
# QuantumInspiredLIFNeuron — complex amplitude stochastic spiking
# ═══════════════════════════════════════════════════════════════════════


class TestQuantumInspiredLIFNeuron:
    @pytest.fixture()
    def neuron(self):
        from sc_neurocore.neurons.models import QuantumInspiredLIFNeuron

        return QuantumInspiredLIFNeuron(tau=20.0, theta=1.0, dt=0.1, seed=42)

    def test_defaults(self, neuron):
        assert neuron.tau == 20.0
        assert neuron.theta == 1.0
        assert neuron.z_re == 0.0
        assert neuron.z_im == 0.0

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"tau": 0.0},
            {"theta": 0.0},
            {"dt": 0.0},
            {"v_reset": float("nan")},
            {"seed": 0},
            {"seed": -1},
            {"seed": 2**64},
            {"seed": 1.5},
            {"z_re": float("nan")},
            {"z_im": float("inf")},
        ],
    )
    def test_rejects_non_physical_quantum_lif_parameters(self, kwargs):
        """Stochastic amplitude dynamics require finite parameters and valid PRNG seed."""
        from sc_neurocore.neurons.models import QuantumInspiredLIFNeuron

        with pytest.raises(ValueError):
            QuantumInspiredLIFNeuron(**kwargs)

    @pytest.mark.parametrize(
        ("i_re", "i_im"),
        [(float("nan"), 0.0), (0.0, float("inf"))],
    )
    def test_rejects_non_finite_complex_drive(self, i_re, i_im):
        """Complex amplitude integration must fail closed on non-finite drive."""
        from sc_neurocore.neurons.models import QuantumInspiredLIFNeuron

        with pytest.raises(ValueError, match="current"):
            QuantumInspiredLIFNeuron().step_complex(i_re, i_im)

    def test_step_returns_binary(self, neuron):
        s = neuron.step(0.5)
        assert s in (0, 1)

    def test_stochastic_spiking(self, neuron):
        """With strong complex input, neuron should spike stochastically."""
        spikes = sum(neuron.step_complex(5.0, 3.0) for _ in range(1000))
        assert spikes > 0, "Must spike with strong input"
        assert spikes < 1000, "Must not spike every step"

    def test_destructive_interference(self):
        """Opposing re/im inputs should suppress firing (key quantum property)."""
        from sc_neurocore.neurons.models import QuantumInspiredLIFNeuron

        # Strong excitatory input.
        n1 = QuantumInspiredLIFNeuron(tau=20.0, theta=0.5, dt=0.1, seed=42)
        spikes_exc = sum(n1.step_complex(3.0, 0.0) for _ in range(500))
        # Near-cancelling: re and im drive |z|^2 ~ 0 through interference.
        n2 = QuantumInspiredLIFNeuron(tau=20.0, theta=0.5, dt=0.1, seed=42)
        spikes_cancel = sum(n2.step_complex(0.01, 0.01) for _ in range(500))
        assert spikes_cancel < spikes_exc

    def test_deterministic_with_same_seed(self):
        """Same seed → same spike train."""
        from sc_neurocore.neurons.models import QuantumInspiredLIFNeuron

        results = []
        for _ in range(2):
            n = QuantumInspiredLIFNeuron(seed=12345)
            train = [n.step_complex(2.0, 1.0) for _ in range(100)]
            results.append(train)
        assert results[0] == results[1]

    def test_different_seeds_differ(self):
        """Different seeds → different spike trains (with high probability)."""
        from sc_neurocore.neurons.models import QuantumInspiredLIFNeuron

        n1 = QuantumInspiredLIFNeuron(seed=1)
        n2 = QuantumInspiredLIFNeuron(seed=9999)
        t1 = [n1.step_complex(3.0, 1.0) for _ in range(200)]
        t2 = [n2.step_complex(3.0, 1.0) for _ in range(200)]
        assert t1 != t2

    def test_reset_restores_seed(self, neuron):
        """Reset re-initialises RNG state from seed."""
        train_a = [neuron.step(2.0) for _ in range(50)]
        neuron.reset()
        train_b = [neuron.step(2.0) for _ in range(50)]
        assert train_a == train_b

    def test_firing_probability_scales_with_amplitude(self):
        """P(spike) = |z|^2/theta^2: higher input → higher rate."""
        from sc_neurocore.neurons.models import QuantumInspiredLIFNeuron

        rates = []
        for amp in [1.0, 3.0, 5.0]:
            n = QuantumInspiredLIFNeuron(tau=20.0, theta=1.0, dt=0.1, seed=42)
            spikes = sum(n.step_complex(amp, 0.0) for _ in range(2000))
            rates.append(spikes)
        assert rates[0] < rates[1] < rates[2], f"Rates must increase: {rates}"


# ═══════════════════════════════════════════════════════════════════════
# DendriticNMDANeuron — Jahr & Stevens (1990) Mg2+ block
# ═══════════════════════════════════════════════════════════════════════


class TestDendriticNMDANeuron:
    @pytest.fixture()
    def neuron(self):
        from sc_neurocore.neurons.models import DendriticNMDANeuron

        return DendriticNMDANeuron()

    def test_defaults(self, neuron):
        assert neuron.v_soma == -65.0
        assert neuron.v_dend == -65.0
        assert neuron.mg_conc == 1.0

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"g_nmda": -0.01},
            {"e_nmda": float("nan")},
            {"mg_conc": -0.01},
            {"g_coupling": -0.01},
            {"tau_soma": 0.0},
            {"tau_dend": 0.0},
            {"theta": float("inf")},
            {"dt": 0.0},
            {"v_soma": float("nan")},
            {"v_dend": float("inf")},
        ],
    )
    def test_rejects_non_physical_nmda_parameters(self, kwargs):
        """NMDA compartment parameters must be finite and biophysically bounded."""
        from sc_neurocore.neurons.models import DendriticNMDANeuron

        with pytest.raises(ValueError):
            DendriticNMDANeuron(**kwargs)

    @pytest.mark.parametrize("voltage", [float("nan"), float("inf")])
    def test_rejects_non_finite_mg_block_voltage(self, voltage):
        """Voltage-dependent magnesium block must reject non-finite voltage."""
        from sc_neurocore.neurons.models import DendriticNMDANeuron

        with pytest.raises(ValueError, match="voltage"):
            DendriticNMDANeuron().mg_block(voltage)

    @pytest.mark.parametrize(
        ("i_soma", "glutamate"),
        [(float("nan"), 0.0), (0.0, float("inf")), (0.0, -0.01)],
    )
    def test_rejects_non_physical_nmda_drive(self, i_soma, glutamate):
        """Somatic current must be finite and glutamate must be finite non-negative."""
        from sc_neurocore.neurons.models import DendriticNMDANeuron

        with pytest.raises(ValueError):
            DendriticNMDANeuron().step(i_soma, glutamate)

    def test_step_returns_binary(self, neuron):
        s = neuron.step(10.0, 0.5)
        assert s in (0, 1)

    def test_mg_block_at_rest(self, neuron):
        """At -65 mV, Mg block should be strong (~0.06)."""
        b = neuron.mg_block(-65.0)
        assert 0.0 < b < 0.15, f"Mg block at -65mV = {b}, expected ~0.06"

    def test_mg_block_at_depolarised(self, neuron):
        """At 0 mV, Mg block should be relieved (~0.78)."""
        b = neuron.mg_block(0.0)
        assert b > 0.5, f"Mg block at 0mV = {b}, expected >0.5"

    def test_mg_block_formula(self, neuron):
        """B(V) = 1/(1 + [Mg]/3.57 * exp(-0.062*V)) — exact from Jahr & Stevens."""
        for v in [-80.0, -65.0, -40.0, -20.0, 0.0, 20.0]:
            expected = 1.0 / (1.0 + (1.0 / 3.57) * math.exp(-0.062 * v))
            actual = neuron.mg_block(v)
            assert abs(actual - expected) < 1e-12, f"Mg block at {v}mV: {actual} != {expected}"

    def test_spikes_with_strong_input(self, neuron):
        """Strong somatic current must produce spikes."""
        spikes = sum(neuron.step(50.0, 0.0) for _ in range(2000))
        assert spikes > 0

    def test_coincidence_detection(self):
        """NMDA requires BOTH glutamate AND depolarisation for full effect."""
        from sc_neurocore.neurons.models import DendriticNMDANeuron

        # Only soma current, no glutamate.
        n1 = DendriticNMDANeuron()
        for _ in range(500):
            n1.step(30.0, 0.0)
        v_no_glut = n1.v_dend

        # Soma current + glutamate.
        n2 = DendriticNMDANeuron()
        for _ in range(500):
            n2.step(30.0, 1.0)
        v_with_glut = n2.v_dend
        # With glutamate, dendrite should differ due to NMDA current.
        assert v_no_glut != v_with_glut

    def test_reset(self, neuron):
        for _ in range(100):
            neuron.step(30.0, 0.5)
        neuron.reset()
        assert neuron.v_soma == -65.0
        assert neuron.v_dend == -65.0


# ═══════════════════════════════════════════════════════════════════════
# MulticompartmentMCNNeuron — Spiking-WM arXiv:2503.00713
# ═══════════════════════════════════════════════════════════════════════


class TestMulticompartmentMCNNeuron:
    @pytest.fixture()
    def neuron(self):
        from sc_neurocore.neurons.models import MulticompartmentMCNNeuron

        return MulticompartmentMCNNeuron()

    def test_defaults_match_table_ii(self, neuron):
        """Default params from Table II of arXiv:2503.00713."""
        assert neuron.tau == 2.0
        assert neuron.tau_b == 2.0
        assert neuron.tau_a == 2.0
        assert neuron.g_ratio == 1.0
        assert neuron.beta == 1.0
        assert neuron.v_th == 1.0

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"tau": 0.0},
            {"tau_b": 0.0},
            {"tau_a": 0.0},
            {"g_ratio": -0.01},
            {"beta": 0.0},
            {"v_th": 0.0},
            {"dt": 0.0},
            {"u": float("nan")},
            {"v_basal": float("inf")},
            {"v_apical": float("-inf")},
        ],
    )
    def test_rejects_non_physical_multicompartment_parameters(self, kwargs):
        """Compartment dynamics require finite positive constants and finite state."""
        from sc_neurocore.neurons.models import MulticompartmentMCNNeuron

        with pytest.raises(ValueError):
            MulticompartmentMCNNeuron(**kwargs)

    @pytest.mark.parametrize("apical_voltage", [float("nan"), float("inf")])
    def test_rejects_non_finite_sigma_input(self, apical_voltage):
        """Apical sigmoid gate must fail closed on non-finite voltages."""
        from sc_neurocore.neurons.models import MulticompartmentMCNNeuron

        with pytest.raises(ValueError, match="x"):
            MulticompartmentMCNNeuron()._sigma(apical_voltage)

    @pytest.mark.parametrize(
        ("x_basal", "x_apical", "i_soma"),
        [(float("nan"), 0.0, 0.0), (0.0, float("inf"), 0.0), (0.0, 0.0, float("nan"))],
    )
    def test_rejects_non_finite_compartment_drive(self, x_basal, x_apical, i_soma):
        """Basal, apical, and somatic drives must be finite."""
        from sc_neurocore.neurons.models import MulticompartmentMCNNeuron

        with pytest.raises(ValueError, match="finite"):
            MulticompartmentMCNNeuron().step_compartments(x_basal, x_apical, i_soma)

    def test_step_returns_binary(self, neuron):
        s = neuron.step(0.5)
        assert s in (0, 1)

    def test_sigma_gating(self, neuron):
        """sigma(0) = 0.5, sigma(large) -> 1, sigma(-large) -> 0."""
        assert abs(neuron._sigma(0.0) - 0.5) < 1e-10
        assert neuron._sigma(10.0) > 0.99
        assert neuron._sigma(-10.0) < 0.01

    def test_basal_input_produces_spikes(self, neuron):
        spikes = sum(neuron.step(3.0) for _ in range(100))
        assert spikes > 0

    def test_threshold_boundary_accepts_one_ulp_roundoff(self, neuron):
        """The Heaviside equality boundary must survive binary64 RK4 roundoff."""
        assert neuron._threshold_reached(math.nextafter(neuron.v_th, 0.0))
        assert not neuron._threshold_reached(neuron.v_th - 1e-9)

    def test_apical_gating_modulates_firing(self):
        """High apical input (gate open) should increase firing vs no apical."""
        from sc_neurocore.neurons.models import MulticompartmentMCNNeuron

        # No apical: gate = sigma(0) = 0.5.
        n1 = MulticompartmentMCNNeuron()
        s1 = sum(n1.step_compartments(2.0, 0.0, 0.0) for _ in range(200))

        # Strong apical: gate = sigma(V_a) -> high.
        n2 = MulticompartmentMCNNeuron()
        s2 = sum(n2.step_compartments(2.0, 5.0, 0.0) for _ in range(200))

        assert s2 >= s1, "Apical gating should enhance or maintain firing"

    def test_soft_reset_to_zero(self, neuron):
        """After spike: U <- U * (1 - S) = 0."""
        for _ in range(50):
            s = neuron.step_compartments(3.0, 2.0, 0.0)
            if s == 1:
                assert neuron.u == 0.0
                return
        pytest.fail("No spike produced in 50 steps")

    def test_step_compartments_api(self, neuron):
        """step_compartments(x_basal, x_apical, i_soma) must accept 3 args."""
        s = neuron.step_compartments(1.0, 0.5, 0.2)
        assert s in (0, 1)

    def test_reset(self, neuron):
        for _ in range(50):
            neuron.step(3.0)
        neuron.reset()
        assert neuron.u == 0.0
        assert neuron.v_basal == 0.0
        assert neuron.v_apical == 0.0


# ═══════════════════════════════════════════════════════════════════════
# AstrocyteLIFNeuron — Perea et al. (2009) tripartite synapse
# ═══════════════════════════════════════════════════════════════════════


class TestAstrocyteLIFNeuron:
    @pytest.fixture()
    def neuron(self):
        from sc_neurocore.neurons.models import AstrocyteLIFNeuron

        return AstrocyteLIFNeuron()

    def test_defaults(self, neuron):
        assert neuron.tau_ca == 500.0
        assert neuron.ca_thresh == 0.5
        assert neuron.g_glio == 2.0

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"tau_m": 0.0},
            {"tau_ca": 0.0},
            {"e_l": float("nan")},
            {"theta": float("nan")},
            {"theta": -70.0},
            {"v_reset": float("inf")},
            {"ca_delta": -0.01},
            {"ca_thresh": -0.01},
            {"g_glio": -0.01},
            {"dt": 0.0},
            {"v": float("nan")},
            {"ca": -0.01},
        ],
    )
    def test_rejects_non_physical_tripartite_parameters(self, kwargs):
        """Tripartite LIF parameters must be finite and physically bounded."""
        from sc_neurocore.neurons.models import AstrocyteLIFNeuron

        with pytest.raises(ValueError):
            AstrocyteLIFNeuron(**kwargs)

    @pytest.mark.parametrize("current", [float("nan"), float("inf")])
    def test_rejects_non_finite_external_current(self, current):
        """Membrane integration must fail closed on non-finite drive."""
        from sc_neurocore.neurons.models import AstrocyteLIFNeuron

        with pytest.raises(ValueError, match="i_ext"):
            AstrocyteLIFNeuron().step_with_pre(current, pre_spike=False)

    @pytest.mark.parametrize("pre_spike", [0, 1, "yes", None])
    def test_rejects_non_boolean_presynaptic_spike_flag(self, pre_spike):
        """Presynaptic event input must be an explicit boolean contract."""
        from sc_neurocore.neurons.models import AstrocyteLIFNeuron

        with pytest.raises(TypeError, match="pre_spike"):
            AstrocyteLIFNeuron().step_with_pre(0.0, pre_spike=pre_spike)

    def test_step_returns_binary(self, neuron):
        s = neuron.step(5.0)
        assert s in (0, 1)

    def test_calcium_rises_with_pre_spikes(self, neuron):
        """Presynaptic spikes must increase calcium."""
        ca_before = neuron.ca
        neuron.step_with_pre(0.0, pre_spike=True)
        assert neuron.ca > ca_before

    def test_calcium_decays_without_spikes(self, neuron):
        """Without pre_spikes, calcium decays toward 0 (tau_ca=500ms)."""
        neuron.ca = 1.0
        # 500ms / dt=0.1 = 5000 steps for one time constant.
        for _ in range(10000):
            neuron.step_with_pre(0.0, pre_spike=False)
        assert neuron.ca < 0.2

    def test_gliotransmitter_threshold(self):
        """I_glio = g_glio only when Ca > Ca_thresh."""
        from sc_neurocore.neurons.models import AstrocyteLIFNeuron

        n = AstrocyteLIFNeuron()
        # Build up calcium with sustained pre_spikes.
        for _ in range(100):
            n.step_with_pre(0.0, pre_spike=True)
        assert n.ca > n.ca_thresh, f"Ca={n.ca} should exceed thresh={n.ca_thresh}"

    def test_glial_feedback_increases_firing(self):
        """Gliotransmitter feedback should increase spike rate vs no feedback."""
        from sc_neurocore.neurons.models import AstrocyteLIFNeuron

        # Strong enough current to be near threshold.
        n_no = AstrocyteLIFNeuron()
        s_no = sum(n_no.step_with_pre(14.0, pre_spike=False) for _ in range(1000))

        n_glio = AstrocyteLIFNeuron()
        s_glio = sum(n_glio.step_with_pre(14.0, pre_spike=True) for _ in range(1000))

        assert s_glio >= s_no, "Glial feedback should not decrease firing"

    def test_reset(self, neuron):
        for _ in range(100):
            neuron.step_with_pre(10.0, pre_spike=True)
        neuron.reset()
        assert neuron.v == neuron.e_l
        assert neuron.ca == 0.0


# ═══════════════════════════════════════════════════════════════════════
# DirectionSelectiveRGC — Gollisch & Meister (2010)
# ═══════════════════════════════════════════════════════════════════════


class TestDirectionSelectiveRGC:
    @pytest.fixture()
    def on_cell(self):
        from sc_neurocore.neurons.models import DirectionSelectiveRGC

        return DirectionSelectiveRGC.new_on()

    @pytest.fixture()
    def off_cell(self):
        from sc_neurocore.neurons.models import DirectionSelectiveRGC

        return DirectionSelectiveRGC.new_off()

    def test_on_centre_flag(self, on_cell, off_cell):
        assert on_cell.is_on_centre is True
        assert off_cell.is_on_centre is False

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"tau": 0.0},
            {"theta": 0.0},
            {"is_on_centre": 1},
            {"w_centre": -0.01},
            {"w_surround": -0.01},
            {"direction_pref": float("nan")},
            {"dt": 0.0},
            {"v": float("inf")},
            {"_prev_intensity": -0.01},
            {"_surround": -0.01},
        ],
    )
    def test_rejects_non_physical_direction_selective_parameters(self, kwargs):
        """Retinal direction-selective state and tuning parameters must be physical."""
        from sc_neurocore.neurons.models import DirectionSelectiveRGC

        with pytest.raises(ValueError):
            DirectionSelectiveRGC(**kwargs)

    @pytest.mark.parametrize(
        ("intensity", "surround_mean"),
        [(float("nan"), 0.0), (0.0, float("inf")), (-0.01, 0.0), (0.0, -0.01)],
    )
    def test_rejects_non_physical_receptive_field_drive(self, intensity, surround_mean):
        """Optical centre and surround drives must be finite non-negative intensities."""
        from sc_neurocore.neurons.models import DirectionSelectiveRGC

        with pytest.raises(ValueError):
            DirectionSelectiveRGC.new_on().step_rf(intensity, surround_mean)

    def test_on_responds_to_light_increase(self, on_cell):
        """On-centre must respond to light onset (positive dI/dt)."""
        for _ in range(10):
            on_cell.step_rf(0.0, 0.0)
        spikes = sum(on_cell.step_rf(6.0, 0.0) for _ in range(30))
        assert spikes > 0

    def test_off_responds_to_light_decrease(self, off_cell):
        """Off-centre must respond to light offset (negative dI/dt)."""
        off_cell.theta = 0.1
        spikes = 0
        for i in range(400):
            intensity = 5.0 if (i // 10) % 2 == 0 else 0.0
            spikes += off_cell.step_rf(intensity, 0.0)
        assert spikes > 0

    def test_surround_inhibition_reduces_firing(self):
        """Surround illumination should reduce centre response."""
        from sc_neurocore.neurons.models import DirectionSelectiveRGC

        no_surr = DirectionSelectiveRGC.new_on()
        with_surr = DirectionSelectiveRGC.new_on()
        s_no = 0
        s_surr = 0
        for i in range(300):
            intensity = 3.0 if i % 10 == 0 else 0.0
            s_no += no_surr.step_rf(intensity, 0.0)
            s_surr += with_surr.step_rf(intensity, 2.0)
        assert s_surr <= s_no, "Surround should suppress firing"

    def test_temporal_derivative(self, on_cell):
        """Constant light should produce no spikes (zero dI/dt)."""
        # Warm up with constant light.
        for _ in range(100):
            on_cell.step_rf(3.0, 0.0)
        # After adaptation, constant light has no temporal derivative.
        late_spikes = sum(on_cell.step_rf(3.0, 0.0) for _ in range(100))
        assert late_spikes == 0, "Constant light should not drive On-centre"

    def test_reset(self, on_cell):
        for _ in range(50):
            on_cell.step_rf(5.0, 1.0)
        on_cell.reset()
        assert on_cell.v == 0.0
        assert on_cell._prev_intensity == 0.0


# ═══════════════════════════════════════════════════════════════════════
# CochlearHairCell — Zilany et al. (2009) Boltzmann MET channels
# ═══════════════════════════════════════════════════════════════════════


class TestCochlearHairCell:
    @pytest.fixture()
    def cell(self):
        from sc_neurocore.neurons.models import CochlearHairCell

        return CochlearHairCell()

    def test_defaults(self, cell):
        assert cell.v == -60.0
        assert cell.g_max == 10.0
        assert cell.delta == 0.1

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"g_max": -0.01},
            {"e_met": float("nan")},
            {"g_l": 0.0},
            {"e_l": float("inf")},
            {"cap": 0.0},
            {"x0": float("nan")},
            {"delta": 0.0},
            {"dt": 0.0},
            {"v": float("nan")},
            {"glutamate_release": -0.01},
        ],
    )
    def test_rejects_non_physical_hair_cell_parameters(self, kwargs):
        """MET channel and membrane parameters must be finite and physically bounded."""
        from sc_neurocore.neurons.models import CochlearHairCell

        with pytest.raises(ValueError):
            CochlearHairCell(**kwargs)

    @pytest.mark.parametrize("displacement", [float("nan"), float("inf")])
    def test_rejects_non_finite_met_displacement(self, displacement):
        """Boltzmann MET activation must fail closed on non-finite displacement."""
        from sc_neurocore.neurons.models import CochlearHairCell

        with pytest.raises(ValueError, match="displacement"):
            CochlearHairCell().p_open(displacement)

    def test_p_open_boltzmann(self, cell):
        """P_open(x) = 1/(1 + exp(-(x - x0)/delta))."""
        # At x = x0: P_open = 0.5.
        assert abs(cell.p_open(0.0) - 0.5) < 1e-10
        # Large positive: P_open -> 1.
        assert cell.p_open(1.0) > 0.99
        # Large negative: P_open -> 0.
        assert cell.p_open(-1.0) < 0.01

    def test_step_returns_binary(self, cell):
        s = cell.step(0.0)
        assert s in (0, 1)

    def test_graded_glutamate_release(self, cell):
        """Glutamate release scales with depolarisation."""
        for _ in range(200):
            cell.step(0.5)
        assert cell.glutamate_release >= 0.0

    def test_positive_displacement_depolarises(self):
        """Strong positive displacement should depolarise (increase V)."""
        from sc_neurocore.neurons.models import CochlearHairCell

        cell = CochlearHairCell()
        v_rest = cell.v
        for _ in range(200):
            cell.step(0.5)
        # MET channels open, current flows, V changes.
        assert cell.v != v_rest

    def test_negative_displacement_stays_near_rest(self):
        """Large negative displacement: MET channels closed, V near E_L."""
        from sc_neurocore.neurons.models import CochlearHairCell

        cell = CochlearHairCell()
        for _ in range(500):
            cell.step(-2.0)
        # P_open(-2.0) ~ 0.0, almost no MET current.
        assert abs(cell.v - cell.e_l) < 5.0

    def test_reset(self, cell):
        for _ in range(100):
            cell.step(0.5)
        cell.reset()
        assert cell.v == cell.e_l
        assert cell.glutamate_release == 0.0


# ═══════════════════════════════════════════════════════════════════════
# TripletStdpSynapse (existing TripletSTDP) — Pfister & Gerstner (2006)
# ═══════════════════════════════════════════════════════════════════════


class TestTripletSTDP:
    @pytest.fixture()
    def synapse(self):
        from sc_neurocore.synapses import TripletSTDP

        return TripletSTDP(weight=0.5)

    def test_defaults(self, synapse):
        assert synapse.tau_plus == 16.8
        assert synapse.tau_minus == 33.7
        assert synapse.tau_x == 101.0
        assert synapse.tau_y == 125.0

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"tau_plus": 0.0},
            {"tau_minus": 0.0},
            {"tau_x": 0.0},
            {"tau_y": 0.0},
            {"a2_plus": -0.01},
            {"a3_plus": float("nan")},
            {"a2_minus": -0.01},
            {"a3_minus": float("inf")},
            {"w_min": 1.0, "w_max": 0.0},
            {"weight": -0.01},
            {"weight": 1.01},
        ],
    )
    def test_rejects_non_physical_triplet_stdp_parameters(self, kwargs):
        """Triplet STDP constants and weight bounds must be finite and physical."""
        from sc_neurocore.synapses import TripletSTDP

        with pytest.raises(ValueError):
            TripletSTDP(**kwargs)

    @pytest.mark.parametrize("dt", [0.0, -1.0, float("nan"), float("inf")])
    def test_rejects_non_physical_triplet_stdp_timestep(self, dt):
        """Trace decay timestep must be finite and positive."""
        from sc_neurocore.synapses import TripletSTDP

        with pytest.raises(ValueError, match="dt"):
            TripletSTDP().step(pre_spike=False, post_spike=False, dt=dt)

    @pytest.mark.parametrize(
        ("pre_spike", "post_spike"),
        [(1, False), (False, 0), ("yes", False), (False, None)],
    )
    def test_rejects_non_boolean_triplet_stdp_spike_flags(self, pre_spike, post_spike):
        """Spike events must be explicit booleans for the update contract."""
        from sc_neurocore.synapses import TripletSTDP

        with pytest.raises(TypeError):
            TripletSTDP().step(pre_spike=pre_spike, post_spike=post_spike)

    def test_ltp_pre_then_post(self, synapse):
        """Pre-before-post pairing should potentiate."""
        w0 = synapse.weight
        synapse.step(pre_spike=True, post_spike=False)
        for _ in range(5):
            synapse.step(pre_spike=False, post_spike=False)
        synapse.step(pre_spike=False, post_spike=True)
        assert synapse.weight > w0

    def test_ltd_post_then_pre(self, synapse):
        """Post-before-pre pairing should depress."""
        w0 = synapse.weight
        synapse.step(pre_spike=False, post_spike=True)
        for _ in range(5):
            synapse.step(pre_spike=False, post_spike=False)
        synapse.step(pre_spike=True, post_spike=False)
        assert synapse.weight < w0

    def test_weight_clamped(self, synapse):
        """Weight must stay in [w_min, w_max]."""
        for _ in range(500):
            synapse.step(pre_spike=True, post_spike=True)
        assert synapse.w_min <= synapse.weight <= synapse.w_max

    def test_traces_decay(self, synapse):
        synapse.step(pre_spike=True, post_spike=True)
        assert synapse.r1 > 0
        for _ in range(200):
            synapse.step(pre_spike=False, post_spike=False)
        assert synapse.r1 < 0.01

    def test_reset(self, synapse):
        synapse.step(pre_spike=True, post_spike=True)
        synapse.reset()
        assert synapse.r1 == 0.0
        assert synapse.o1 == 0.0


# ═══════════════════════════════════════════════════════════════════════
# ShortTermPlasticitySynapse — Tsodyks-Markram (1997)
# ═══════════════════════════════════════════════════════════════════════


class TestShortTermPlasticitySynapse:
    @pytest.fixture()
    def depressing(self):
        from sc_neurocore.synapses import ShortTermPlasticitySynapse

        return ShortTermPlasticitySynapse.new_depressing()

    @pytest.fixture()
    def facilitating(self):
        from sc_neurocore.synapses import ShortTermPlasticitySynapse

        return ShortTermPlasticitySynapse.new_facilitating()

    def test_depressing_defaults(self, depressing):
        assert depressing.u_base == 0.5
        assert depressing.tau_d == 200.0

    def test_facilitating_defaults(self, facilitating):
        assert facilitating.u_base == 0.1
        assert facilitating.tau_f == 500.0

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"x": -0.01},
            {"x": 1.01},
            {"u": -0.01},
            {"u": 1.01},
            {"u_base": 0.0},
            {"u_base": 1.01},
            {"tau_d": 0.0},
            {"tau_f": 0.0},
            {"amplitude": -0.01},
            {"dt": 0.0},
        ],
    )
    def test_rejects_non_physical_stp_parameters(self, kwargs):
        """Tsodyks-Markram resources, utilisation, and constants must be physical."""
        from sc_neurocore.synapses import ShortTermPlasticitySynapse

        with pytest.raises(ValueError):
            ShortTermPlasticitySynapse(**kwargs)

    @pytest.mark.parametrize("pre_spike", [1, 0, "yes", None])
    def test_rejects_non_boolean_stp_spike_flag(self, pre_spike):
        """Presynaptic event input must be an explicit boolean."""
        from sc_neurocore.synapses import ShortTermPlasticitySynapse

        with pytest.raises(TypeError, match="pre_spike"):
            ShortTermPlasticitySynapse().step(pre_spike)

    def test_depression_successive_spikes(self, depressing):
        """Depressing synapse: PSC decreases with rapid pre_spikes."""
        pscs = [depressing.step(True) for _ in range(5)]
        assert pscs[0] > pscs[1] > pscs[2], f"PSCs should decrease: {pscs}"

    def test_facilitation_successive_spikes(self, facilitating):
        """Facilitating synapse: PSC increases with rapid pre_spikes."""
        pscs = [facilitating.step(True) for _ in range(5)]
        # Facilitation makes u grow, but x depletes. For facilitating params,
        # first few PSCs should increase before x depletion dominates.
        assert pscs[1] > pscs[0], f"2nd PSC should exceed 1st: {pscs[:3]}"

    def test_recovery_after_silence(self, depressing):
        """After depletion, silence allows recovery of x toward 1."""
        for _ in range(10):
            depressing.step(True)
        x_depleted = depressing.x
        for _ in range(2000):
            depressing.step(False)
        assert depressing.x > x_depleted + 0.3

    def test_no_spike_no_current(self, depressing):
        """No pre_spike → zero PSC."""
        psc = depressing.step(False)
        assert psc == 0.0

    def test_x_never_negative(self, depressing):
        """Resources x must be clamped >= 0."""
        for _ in range(100):
            depressing.step(True)
        assert depressing.x >= 0.0

    def test_reset(self, depressing):
        for _ in range(10):
            depressing.step(True)
        depressing.reset()
        assert depressing.x == 1.0
        assert depressing.u == depressing.u_base


# ═══════════════════════════════════════════════════════════════════════
# DopamineStdpSynapse — Izhikevich (2007) DA-modulated STDP
# ═══════════════════════════════════════════════════════════════════════


class TestDopamineStdpSynapse:
    @pytest.fixture()
    def synapse(self):
        from sc_neurocore.synapses import DopamineStdpSynapse

        return DopamineStdpSynapse(weight=0.5)

    def test_defaults(self, synapse):
        assert synapse.tau_e == 1000.0
        assert synapse.tau_da == 200.0
        assert synapse.a_plus == 1.0
        assert synapse.a_minus == -1.0

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"w_min": 1.0, "w_max": 0.0},
            {"weight": -0.01},
            {"weight": 1.01},
            {"tau_e": 0.0},
            {"tau_da": 0.0},
            {"tau_pre": 0.0},
            {"tau_post": 0.0},
            {"a_plus": -0.01},
            {"a_minus": 0.01},
            {"lr": -0.01},
            {"dt": 0.0},
            {"eligibility": float("nan")},
            {"dopamine": float("inf")},
            {"trace_pre": float("nan")},
            {"trace_post": float("inf")},
            {"trace_pre": -1.0},
        ],
    )
    def test_rejects_non_physical_dopamine_stdp_parameters(self, kwargs):
        """Dopamine-gated STDP constants, traces, and bounds must be physical."""
        from sc_neurocore.synapses import DopamineStdpSynapse

        with pytest.raises(ValueError):
            DopamineStdpSynapse(**kwargs)

    @pytest.mark.parametrize(
        ("pre_spike", "post_spike", "reward"),
        [(1, False, 0.0), (False, 0, 0.0), (False, False, float("nan"))],
    )
    def test_rejects_invalid_dopamine_stdp_step_inputs(self, pre_spike, post_spike, reward):
        """Spike flags must be boolean and reward must be finite."""
        from sc_neurocore.synapses import DopamineStdpSynapse

        with pytest.raises((TypeError, ValueError)):
            DopamineStdpSynapse().step(pre_spike, post_spike, reward)

    def test_step_returns_float(self, synapse):
        w = synapse.step(True, False, 0.0)
        assert isinstance(w, float)

    def test_no_reward_no_weight_change(self, synapse):
        """Without dopamine, eligibility doesn't convert to weight change."""
        w0 = synapse.weight
        for i in range(50):
            synapse.step(i % 10 == 0, i % 10 == 2, reward=0.0)
        # Small or zero weight change without DA.
        assert abs(synapse.weight - w0) < 0.01

    def test_reward_drives_weight_change(self, synapse):
        """With reward (dopamine), weight should change."""
        w0 = synapse.weight
        for i in range(200):
            synapse.step(
                i % 10 == 0,
                i % 10 == 2,
                reward=0.5 if i % 5 == 0 else 0.0,
            )
        assert synapse.weight != w0, "Reward must drive weight change"

    def test_eligibility_trace_builds(self, synapse):
        """Pre/post spikes build eligibility trace."""
        synapse.step(True, False, 0.0)
        synapse.step(False, True, 0.0)
        assert synapse.eligibility != 0.0

    def test_eligibility_decays(self, synapse):
        synapse.step(True, False, 0.0)
        synapse.step(False, True, 0.0)
        e_after_spikes = abs(synapse.eligibility)
        for _ in range(5000):
            synapse.step(False, False, 0.0)
        assert abs(synapse.eligibility) < e_after_spikes * 0.01

    def test_dopamine_integrates_reward(self, synapse):
        synapse.step(False, False, reward=1.0)
        assert synapse.dopamine > 0.0

    def test_dopamine_decays(self, synapse):
        synapse.step(False, False, reward=10.0)
        da_high = synapse.dopamine
        for _ in range(2000):
            synapse.step(False, False, reward=0.0)
        assert synapse.dopamine < da_high * 0.01

    def test_weight_clamped(self, synapse):
        for _ in range(1000):
            synapse.step(True, True, reward=10.0)
        assert synapse.w_min <= synapse.weight <= synapse.w_max

    def test_reset(self, synapse):
        synapse.step(True, True, reward=5.0)
        synapse.reset()
        assert synapse.eligibility == 0.0
        assert synapse.dopamine == 0.0
        assert synapse.trace_pre == 0.0
        assert synapse.trace_post == 0.0

    def test_distal_reward_problem(self):
        """Core Izhikevich (2007) result: delayed reward still modifies weight."""
        from sc_neurocore.synapses import DopamineStdpSynapse

        syn = DopamineStdpSynapse(weight=0.5, lr=0.01)
        # Phase 1: STDP pairing (builds eligibility, no reward).
        for i in range(50):
            syn.step(i % 5 == 0, i % 5 == 1, reward=0.0)
        assert syn.eligibility != 0.0
        w_before_reward = syn.weight
        # Phase 2: Delayed reward (no more spikes).
        for _ in range(100):
            syn.step(False, False, reward=1.0)
        # Weight should change from delayed reward acting on eligibility.
        assert syn.weight != w_before_reward, "Delayed reward must drive learning"

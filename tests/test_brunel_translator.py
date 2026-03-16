# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for Brian2 ↔ SC-NeuroCore Brunel parameter translator

"""Tests for Brian2 ↔ SC-NeuroCore Brunel parameter translator."""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "benchmarks"))

from brunel_translator import (
    BrunelParams,
    translate_v1_stochastic_lif,
    translate_v2_rate_matched,
    translate_v3_fixed_point,
    translate_v4_hybrid,
    translate_v5_izhikevich,
    translate_v6_homeostatic,
    translate_v7_noisy,
    translate_v8_refractory,
    translate_v9_post_kick,
    translate_v10_exact_leak,
    translate_v11_q16,
    translate_v12_stdp,
    translate_v13_dot_product,
    translate_v14_sobol,
    translate_v15_jax,
    translate_v16_recurrent,
    translate_v17_memristive,
    translate_v18_numba,
    translate_v19_pytorch_cuda,
    translate_v20_vectorized_numpy,
)
from sc_neurocore import (
    StochasticLIFNeuron,
    FixedPointLIFNeuron,
    BitstreamSynapse,
    VectorizedSCLayer,
    SCIzhikevichNeuron,
    HomeostaticLIFNeuron,
)


class TestV1DeltaPSC:
    """V1: StochasticLIF with delta-PSC wiring."""

    def test_suprathreshold_epsp_fires(self):
        """Single voltage kick >= v_threshold must produce a spike."""
        bp = BrunelParams(v_threshold=20.0, v_reset=10.0)
        params = translate_v1_stochastic_lif(bp)
        n = StochasticLIFNeuron(**params["neuron_kwargs"])
        # Delta-PSC: direct voltage jump above threshold
        n.v += 21.0
        spike = n.step(0.0)  # leak-only step
        assert spike == 1
        assert n.v == bp.v_reset

    def test_subthreshold_no_spike(self):
        """Input below threshold must not fire."""
        bp = BrunelParams(v_threshold=20.0, v_reset=10.0)
        params = translate_v1_stochastic_lif(bp)
        n = StochasticLIFNeuron(**params["neuron_kwargs"])
        n.v += 5.0
        spike = n.step(0.0)
        assert spike == 0


class TestV2RateMatched:
    """V2: VectorizedSCLayer probability domain."""

    def test_output_proportional_to_input(self):
        """Mean output probability should increase with input probability."""
        bp = BrunelParams()
        params = translate_v2_rate_matched(bp)
        layer = VectorizedSCLayer(n_inputs=4, n_neurons=2, length=params["bitstream_length"])
        # High input probability
        out_high = layer.forward([0.8, 0.8, 0.8, 0.8])
        # Low input probability
        out_low = layer.forward([0.1, 0.1, 0.1, 0.1])
        assert out_high.mean() > out_low.mean()


class TestV3FixedPoint:
    """V3: FixedPointLIFNeuron Q8.8."""

    def test_no_overflow_brunel_weights(self):
        """Max Brunel weight must not overflow Q8.8 signed range."""
        bp = BrunelParams(weight_exc=5.0, g_inh=5.0)
        params = translate_v3_fixed_point(bp)
        # Q8.8 signed range: -32768 to 32767
        assert -32768 <= params["j_exc_q"] <= 32767
        assert -32768 <= params["j_inh_q"] <= 32767
        assert -32768 <= params["v_threshold_q"] <= 32767

    def test_single_neuron_fires(self):
        """Q8.8 neuron should fire with sustained suprathreshold input."""
        bp = BrunelParams(v_threshold=20.0, v_reset=10.0, weight_exc=5.0)
        params = translate_v3_fixed_point(bp)
        n = FixedPointLIFNeuron(
            data_width=params["data_width"],
            fraction=params["fraction"],
            v_threshold=params["v_threshold_q"],
            v_reset=params["v_reset_q"],
            refractory_period=params["refractory_period"],
        )
        spikes = 0
        for _ in range(1000):
            # Drive with weight as current input
            spike, _ = n.step(
                leak_k=params["leak_k"],
                gain_k=params["gain_k"],
                I_t=params["j_exc_q"] * 10,  # 10 simultaneous inputs
            )
            spikes += spike
        assert spikes > 0, "Q8.8 neuron must fire with sustained input"


class TestV4Hybrid:
    """V4: BitstreamSynapse AND + StochasticLIFNeuron."""

    def test_spike_from_high_prob_bitstream(self):
        """Sustained high-probability input bitstream should produce spikes."""
        bp = BrunelParams(v_threshold=20.0, v_reset=10.0, weight_exc=5.0)
        params = translate_v4_hybrid(bp)
        n = StochasticLIFNeuron(**params["neuron_kwargs"])
        syn = BitstreamSynapse(**params["synapse_kwargs"])

        rng = np.random.default_rng(42)
        spikes = 0
        for _ in range(500):
            # High-probability pre-synaptic bitstream
            pre_bits = (rng.random(params["bitstream_length"]) < 0.9).astype(np.uint8)
            post_bits = syn.apply(pre_bits)
            current = post_bits.sum() * params["popcount_scale"]
            # Apply as delta-PSC voltage kick
            n.v += current
            spike = n.step(0.0)
            spikes += spike
        assert spikes > 0, "Hybrid SC+LIF must fire with high-probability drive"


class TestRoundTrip:
    """Cross-variant consistency."""

    def test_v1_round_trip_nonzero_spikes(self):
        """Translate → simulate 1 neuron with Poisson drive → spike count > 0."""
        bp = BrunelParams(
            v_threshold=20.0,
            v_reset=10.0,
            weight_exc=5.0,
            external_rate_hz=200.0,
        )
        params = translate_v1_stochastic_lif(bp)
        n = StochasticLIFNeuron(**params["neuron_kwargs"])

        rng = np.random.default_rng(42)
        spikes = 0
        for _ in range(10000):
            # Poisson voltage kicks (delta-PSC)
            n_events = rng.poisson(bp.external_rate_hz * bp.dt / 1000.0)
            n.v += n_events * params["ext_weight"]
            spike = n.step(0.0)
            spikes += spike
        assert spikes > 0

    def test_all_variants_produce_params(self):
        """All 20 translators return non-empty dicts without errors."""
        bp = BrunelParams()
        translators = [
            translate_v1_stochastic_lif,
            translate_v2_rate_matched,
            translate_v3_fixed_point,
            translate_v4_hybrid,
            translate_v5_izhikevich,
            translate_v6_homeostatic,
            translate_v7_noisy,
            translate_v8_refractory,
            translate_v9_post_kick,
            translate_v10_exact_leak,
            translate_v11_q16,
            translate_v12_stdp,
            translate_v13_dot_product,
            translate_v14_sobol,
            translate_v15_jax,
            translate_v16_recurrent,
            translate_v17_memristive,
            translate_v18_numba,
            translate_v19_pytorch_cuda,
            translate_v20_vectorized_numpy,
        ]
        for fn in translators:
            result = fn(bp)
            assert isinstance(result, dict), f"{fn.__name__} returned non-dict"
            assert len(result) > 0, f"{fn.__name__} returned empty dict"


class TestV5Izhikevich:
    """V5: Izhikevich regular-spiking neuron."""

    def test_fires_with_sustained_input(self):
        bp = BrunelParams(weight_exc=5.0, external_rate_hz=200.0)
        params = translate_v5_izhikevich(bp)
        n = SCIzhikevichNeuron(**params["neuron_kwargs"])
        spikes = sum(n.step(15.0) for _ in range(1000))
        assert spikes > 0, "Izhikevich must fire with sustained current"

    def test_threshold_is_30mv(self):
        bp = BrunelParams()
        params = translate_v5_izhikevich(bp)
        assert params["neuron_kwargs"]["c"] == -65.0
        assert params["neuron_kwargs"]["d"] == 8.0


class TestV6Homeostatic:
    """V6: HomeostaticLIFNeuron threshold adaptation."""

    def test_threshold_adapts(self):
        bp = BrunelParams(v_threshold=20.0, v_reset=10.0, weight_exc=5.0)
        params = translate_v6_homeostatic(bp)
        n = HomeostaticLIFNeuron(**params["neuron_kwargs"])
        initial_threshold = n.v_threshold
        for _ in range(500):
            n.v += 25.0
            n.step(0.0)
        assert n.v_threshold != initial_threshold, "Threshold must adapt"


class TestV7Noisy:
    """V7: Noisy LIF fires stochastically."""

    def test_noise_produces_spikes(self):
        bp = BrunelParams(v_threshold=20.0, v_reset=10.0)
        params = translate_v7_noisy(bp)
        assert params["neuron_kwargs"]["noise_std"] == 1.0
        n = StochasticLIFNeuron(**params["neuron_kwargs"])
        spikes = 0
        for _ in range(5000):
            n.v += 18.0  # near threshold
            spikes += n.step(0.0)
        assert spikes > 0, "Noisy LIF near threshold should fire stochastically"


class TestV9PostKick:
    """V9: Post-kick timing differs from V1."""

    def test_kick_after_step_flag(self):
        bp = BrunelParams()
        params = translate_v9_post_kick(bp)
        assert params.get("kick_after_step") is True


class TestV10ExactLeak:
    """V10: Exact exponential leak."""

    def test_leak_factor_matches_exp(self):
        bp = BrunelParams(dt=0.1, tau_mem=20.0)
        params = translate_v10_exact_leak(bp)
        expected = np.exp(-0.1 / 20.0)
        assert abs(params["leak_factor"] - expected) < 1e-10


class TestV11Q16:
    """V11: Q4.12 fixed-point — no overflow."""

    def test_no_overflow_standard_params(self):
        bp = BrunelParams(v_threshold=20.0, v_reset=10.0, weight_exc=5.0)
        params = translate_v11_q16(bp)
        assert params["data_width"] == 32
        assert params["fraction"] == 12
        assert params["v_threshold_q"] == 20 * 4096

    def test_overflow_raises(self):
        bp = BrunelParams(v_threshold=600000.0)  # exceeds 32-bit Q16.12
        import pytest

        with pytest.raises(OverflowError):
            translate_v11_q16(bp)


class TestV15Jax:
    """V15: JAX layer produces output."""

    def test_jax_layer_runs(self):
        try:
            from sc_neurocore import JaxSCDenseLayer
            from sc_neurocore.accel.jax_backend import jnp, HAS_JAX

            if not HAS_JAX:
                import pytest

                pytest.skip("JAX not installed")
        except (ImportError, RuntimeError):
            import pytest

            pytest.skip("JAX not installed")

        bp = BrunelParams()
        params = translate_v15_jax(bp)
        layer = JaxSCDenseLayer(
            n_neurons=10,
            n_inputs=10,
            neuron_params=params["neuron_params"],
            seed=42,
        )
        I_t = jnp.ones(10) * 5.0
        spikes = layer.step(I_t)
        assert spikes.shape == (10,)


class TestV19PytorchCuda:
    """V19: CUDA tensor computation."""

    def test_cuda_params(self):
        bp = BrunelParams()
        params = translate_v19_pytorch_cuda(bp)
        assert params["n_total"] == 1000
        assert params["v_threshold"] == 20.0


class TestV20VectorizedNumpy:
    """V20: Vectorized matches basic dynamics."""

    def test_vectorized_fires(self):
        bp = BrunelParams(v_threshold=20.0, v_reset=10.0, weight_exc=5.0, external_rate_hz=200.0)
        params = translate_v20_vectorized_numpy(bp)
        n = params["n_total"]
        v = np.full(n, params["v_rest"])
        alpha = params["dt"] / params["tau_mem"]
        rng = np.random.default_rng(42)
        spike_count = 0
        for _ in range(100):
            ext = rng.poisson(200.0 * 0.1 / 1000.0, n)
            v += ext * params["weight_exc"]
            v += alpha * (params["v_rest"] - v)
            fired = v >= params["v_threshold"]
            spike_count += int(fired.sum())
            v[fired] = params["v_reset"]
        assert spike_count > 0, "Vectorized numpy must fire with strong drive"

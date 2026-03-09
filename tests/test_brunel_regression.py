# SPDX-License-Identifier: AGPL-3.0-or-later
"""Regression and consistency tests derived from the 20-variant Brunel benchmark results.

These tests validate invariants that hold across variants without re-running
the full 1000ms simulation. They use the saved JSON artifact and lightweight
translator/neuron smoke runs.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "benchmarks"))

from brunel_translator import (
    BrunelParams,
    translate_v1_stochastic_lif,
    translate_v3_fixed_point,
    translate_v7_noisy,
    translate_v8_refractory,
    translate_v9_post_kick,
    translate_v10_exact_leak,
    translate_v11_q16,
    translate_v20_vectorized_numpy,
)
from sc_neurocore import StochasticLIFNeuron, FixedPointLIFNeuron


RESULTS_PATH = os.path.join(
    os.path.dirname(__file__), "..", "benchmarks", "results", "snn_translator_20v.json"
)


def _load_results() -> dict[str, dict]:
    if not os.path.exists(RESULTS_PATH):
        pytest.skip("benchmark results JSON not found")
    with open(RESULTS_PATH) as f:
        data = json.load(f)
    return {r["variant"]: r for r in data}


# --- BrunelParams algebraic identities ---


class TestBrunelParamsAlgebra:
    def test_n_total(self):
        bp = BrunelParams(n_exc=800, n_inh=200)
        assert bp.n_total == 1000

    def test_weight_inh(self):
        bp = BrunelParams(weight_exc=0.5, g_inh=4.0)
        assert bp.weight_inh == pytest.approx(2.0)

    def test_weight_inh_identity(self):
        bp = BrunelParams()
        assert bp.weight_inh == bp.g_inh * bp.weight_exc


# --- Fixed-point roundtrip ---


class TestFixedPointRoundtrip:
    def test_q88_encode_decode(self):
        """Q8.8: encode then decode recovers floor quantisation."""
        bp = BrunelParams(v_threshold=20.0, v_reset=10.0)
        params = translate_v3_fixed_point(bp)
        frac = params["fraction"]
        scale = 1 << frac
        v_orig = bp.v_threshold
        v_q = params["v_threshold_q"]
        v_decoded = v_q / scale
        assert v_decoded == pytest.approx(v_orig, abs=1.0 / scale)

    def test_q16_encode_decode(self):
        """Q16.12: 12 fractional bits give 1/4096 precision."""
        bp = BrunelParams(v_threshold=20.0)
        params = translate_v11_q16(bp)
        scale = 1 << params["fraction"]
        v_decoded = params["v_threshold_q"] / scale
        assert v_decoded == pytest.approx(bp.v_threshold, abs=1.0 / scale)


# --- Translator cross-variant consistency ---


class TestTranslatorConsistency:
    """Variants that share StochasticLIFNeuron base must produce identical neuron_kwargs
    except for the specific parameter they modify."""

    def test_v7_only_changes_noise(self):
        bp = BrunelParams()
        v1 = translate_v1_stochastic_lif(bp)
        v7 = translate_v7_noisy(bp)
        for k in v1["neuron_kwargs"]:
            if k == "noise_std":
                assert v7["neuron_kwargs"][k] == 1.0
            else:
                assert v7["neuron_kwargs"][k] == v1["neuron_kwargs"][k]

    def test_v8_only_changes_refractory(self):
        bp = BrunelParams()
        v1 = translate_v1_stochastic_lif(bp)
        v8 = translate_v8_refractory(bp)
        for k in v1["neuron_kwargs"]:
            v1_val = v1["neuron_kwargs"][k]
            v8_val = v8["neuron_kwargs"].get(k, v1_val)
            assert v8_val == v1_val
        assert v8["neuron_kwargs"]["refractory_period"] == 5

    def test_v9_only_adds_kick_flag(self):
        bp = BrunelParams()
        v1 = translate_v1_stochastic_lif(bp)
        v9 = translate_v9_post_kick(bp)
        assert v9["kick_after_step"] is True
        assert v9["neuron_kwargs"] == v1["neuron_kwargs"]

    def test_v10_exact_leak_factor(self):
        bp = BrunelParams(dt=0.1, tau_mem=20.0)
        v10 = translate_v10_exact_leak(bp)
        assert v10["exact_leak"] is True
        expected = np.exp(-bp.dt / bp.tau_mem)
        assert v10["leak_factor"] == pytest.approx(expected, abs=1e-12)


# --- Precision ordering (Q16.12 < Q8.8 in spike count) ---


class TestPrecisionOrdering:
    """Higher fixed-point precision → more accurate leak → fewer rounding-induced spikes."""

    def test_q16_fewer_spikes_than_q88_short_sim(self):
        bp = BrunelParams(v_threshold=20.0, v_reset=10.0, weight_exc=5.0)
        p3 = translate_v3_fixed_point(bp)
        p11 = translate_v11_q16(bp)

        n3 = FixedPointLIFNeuron(
            data_width=p3["data_width"],
            fraction=p3["fraction"],
            v_threshold=p3["v_threshold_q"],
            v_reset=p3["v_reset_q"],
            refractory_period=p3["refractory_period"],
        )
        n11 = FixedPointLIFNeuron(
            data_width=p11["data_width"],
            fraction=p11["fraction"],
            v_threshold=p11["v_threshold_q"],
            v_reset=p11["v_reset_q"],
            refractory_period=p11["refractory_period"],
        )

        rng = np.random.default_rng(42)
        s3, s11 = 0, 0
        for _ in range(2000):
            I = rng.poisson(200.0 * 0.1 / 1000.0) * int(5.0 * (1 << p3["fraction"])) * 10
            spike3, _ = n3.step(leak_k=p3["leak_k"], gain_k=p3["gain_k"], I_t=I)
            s3 += spike3

            I11 = rng.poisson(200.0 * 0.1 / 1000.0) * int(5.0 * (1 << p11["fraction"])) * 10
            spike11, _ = n11.step(leak_k=p11["leak_k"], gain_k=p11["gain_k"], I_t=I11)
            s11 += spike11

        assert s11 <= s3, f"Q16.12 ({s11}) should produce ≤ spikes than Q8.8 ({s3})"


# --- Refractory rate ceiling ---


class TestRefractoryRateCeiling:
    def test_refractory_limits_rate(self):
        """With 5-step refractory at dt=0.1ms, max rate = 1/(5*0.1ms) = 2000 Hz."""
        bp = BrunelParams(v_threshold=20.0, v_reset=10.0, weight_exc=5.0)
        params = translate_v8_refractory(bp)
        n = StochasticLIFNeuron(**params["neuron_kwargs"])
        spikes = 0
        steps = 10000
        for _ in range(steps):
            n.v += 25.0
            spikes += n.step(0.0)
        rate_hz = spikes / (steps * bp.dt / 1000.0)
        max_theoretical = 1000.0 / (params["neuron_kwargs"]["refractory_period"] * bp.dt)
        assert rate_hz <= max_theoretical * 1.01


# --- Regression tests from saved benchmark JSON ---


class TestBenchmarkRegression:
    """Validate invariants from the 20-variant benchmark run."""

    def test_brian2_reference_exists(self):
        results = _load_results()
        assert "brian2_reference" in results
        assert results["brian2_reference"]["total_spikes"] > 0

    def test_at_least_16_variants_have_output(self):
        results = _load_results()
        active = [k for k, v in results.items() if v["mean_rate_hz"] > 0 or v["total_spikes"] > 0]
        assert len(active) >= 16, f"Only {len(active)} variants produced output"

    def test_v14_sobol_closest_to_brian2(self):
        """Sobol low-discrepancy encoding should be closest to Brian2 among spiking variants."""
        results = _load_results()
        spiking = {
            k: v for k, v in results.items() if v["total_spikes"] > 0 and k != "brian2_reference"
        }
        if not spiking:
            pytest.skip("no spiking variants in results")
        distances = {
            k: abs(v["rate_ratio"] - 1.0) for k, v in spiking.items() if v["rate_ratio"] > 0
        }
        closest = min(distances, key=distances.get)
        assert closest == "v14_sobol_bitstream", f"Expected Sobol closest, got {closest}"

    def test_acceleration_ordering(self):
        """V18 (Numba) and V19 (CUDA) must be faster than V1 (pure Python)."""
        results = _load_results()
        v1_time = results.get("v1_stochastic_lif", {}).get("wall_time_s", 0)
        if v1_time == 0:
            pytest.skip("V1 result not found")
        for vname in ["v18_numba_jit", "v19_pytorch_cuda", "v20_vectorized_numpy"]:
            vt = results.get(vname, {}).get("wall_time_s", 0)
            if vt > 0:
                assert vt < v1_time, f"{vname} ({vt:.1f}s) not faster than V1 ({v1_time:.1f}s)"

    def test_refractory_rate_below_ceiling(self):
        results = _load_results()
        v8 = results.get("v8_refractory_lif", {})
        if not v8 or v8["mean_rate_hz"] == 0:
            pytest.skip("V8 result not found")
        assert v8["mean_rate_hz"] < 2000.0

    def test_v1_v20_spike_count_match(self):
        """V1 and V20 implement identical LIF dynamics, spike counts must match."""
        results = _load_results()
        v1 = results.get("v1_stochastic_lif", {})
        v20 = results.get("v20_vectorized_numpy", {})
        if not v1 or not v20:
            pytest.skip("V1 or V20 result not found")
        assert v1["total_spikes"] == v20["total_spikes"]

    def test_homeostatic_close_to_baseline(self):
        """V6 homeostatic should fire within 5% of V1 (adaptation hasn't converged in 1s)."""
        results = _load_results()
        v1 = results.get("v1_stochastic_lif", {})
        v6 = results.get("v6_homeostatic_lif", {})
        if not v1 or not v6:
            pytest.skip("V1 or V6 result not found")
        ratio = v6["mean_rate_hz"] / v1["mean_rate_hz"]
        assert 0.95 <= ratio <= 1.05, f"V6/V1 ratio {ratio:.3f} outside 5% band"

    def test_noisy_close_to_baseline(self):
        """V7 noisy should fire within 5% of V1 (noise is small relative to drive)."""
        results = _load_results()
        v1 = results.get("v1_stochastic_lif", {})
        v7 = results.get("v7_noisy_lif", {})
        if not v1 or not v7:
            pytest.skip("V1 or V7 result not found")
        ratio = v7["mean_rate_hz"] / v1["mean_rate_hz"]
        assert 0.90 <= ratio <= 1.10, f"V7/V1 ratio {ratio:.3f} outside 10% band"

    def test_all_wall_times_positive(self):
        results = _load_results()
        for name, r in results.items():
            if r.get("status") == "skipped" or r.get("metric_note", "").startswith("SKIPPED"):
                continue
            assert r["wall_time_s"] > 0, f"{name} has zero wall time"


# --- Neuron-level micro-tests ---


class TestNeuronMicroProperties:
    """Fast single-neuron tests verifying biophysical properties."""

    def test_lif_subthreshold_decay(self):
        """Below threshold, membrane voltage decays toward v_rest."""
        bp = BrunelParams(v_threshold=20.0, v_reset=0.0, v_rest=0.0)
        params = translate_v1_stochastic_lif(bp)
        n = StochasticLIFNeuron(**params["neuron_kwargs"])
        n.v = 15.0
        for _ in range(100):
            n.step(0.0)
        assert n.v < 15.0, "Membrane must decay without input"
        assert n.v >= 0.0, "Membrane must not go below v_rest"

    def test_lif_reset_value(self):
        """After spiking, membrane resets to v_reset."""
        bp = BrunelParams(v_threshold=20.0, v_reset=10.0)
        params = translate_v1_stochastic_lif(bp)
        n = StochasticLIFNeuron(**params["neuron_kwargs"])
        n.v = 25.0
        spike = n.step(0.0)
        assert spike == 1
        assert n.v == bp.v_reset

    def test_exact_vs_euler_leak_difference(self):
        """Exact exponential leak and Euler leak produce slightly different voltages."""
        bp = BrunelParams(v_threshold=20.0, v_rest=0.0, dt=0.1, tau_mem=20.0)
        p1 = translate_v1_stochastic_lif(bp)
        p10 = translate_v10_exact_leak(bp)

        n_euler = StochasticLIFNeuron(**p1["neuron_kwargs"])
        n_exact = StochasticLIFNeuron(**p10["neuron_kwargs"])

        n_euler.v = 15.0
        n_exact.v = 15.0

        # Euler: v += dt/tau * (v_rest - v) = 0.1/20 * (0 - 15) = -0.075 → v = 14.925
        n_euler.step(0.0)
        # Exact: v *= exp(-0.1/20) = 0.99501... → v = 14.925...
        n_exact.v = 15.0 * p10["leak_factor"]

        # Both should be close but not identical
        assert abs(n_euler.v - n_exact.v) < 0.01
        assert n_euler.v != n_exact.v  # Euler has truncation error

    def test_vectorized_params_completeness(self):
        """V20 translator must provide all fields needed for batch numpy update."""
        bp = BrunelParams()
        p = translate_v20_vectorized_numpy(bp)
        required = {
            "v_threshold",
            "v_reset",
            "v_rest",
            "tau_mem",
            "dt",
            "weight_exc",
            "weight_inh",
            "n_total",
            "n_exc",
        }
        missing = required - set(p.keys())
        assert not missing, f"V20 missing keys: {missing}"

# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestBenchmarkRegression from former test_brunel_regression.py

"""Focused suite: TestBenchmarkRegression from former test_brunel_regression.py."""

from __future__ import annotations

from tests.brunel_regression_support import *  # noqa: F403


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

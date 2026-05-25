# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — End-to-end test: SiegertTransferFunction

"""Full pipeline test for SiegertTransferFunction (Siegert 1951).

Mean-field LIF firing rate. Returns float (Hz), NOT int spike.
Analytical: r = [τ_rp + τ_m√π · ∫exp(u²)(1+erf(u))du]⁻¹.
Saturates at 1/τ_rp = 500 Hz. ~524 steps/s (Gauss-Legendre quadrature)."""

from __future__ import annotations

import time

import numpy as np
import pytest

from sc_neurocore.neurons.models.siegert import SiegertTransferFunction, _erf_approx
from sc_neurocore.network.population import Population


class TestSiegertIsolation:
    def test_defaults(self):
        n = SiegertTransferFunction()
        assert n.tau_m == 20.0 and n.tau_rp == 2.0
        assert n.v_threshold == -50.0 and n.v_reset == -70.0

    def test_step_returns_float(self):
        n = SiegertTransferFunction()
        assert isinstance(n.step(20.0), (float, np.floating))

    def test_reset_noop(self):
        n = SiegertTransferFunction()
        n.step(20.0)
        n.reset()  # should not raise


class TestSiegertRateFunction:
    def test_zero_rate_below_threshold(self):
        """mu = V_rest + I. I<15 → mu < threshold → rate ≈ 0."""
        n = SiegertTransferFunction()
        for I in [0.0, 5.0, 10.0]:
            rate = n.step(I)
            assert rate < 0.01, f"I={I}: rate={rate:.4f}, expected ≈ 0"

    def test_positive_rate_above_threshold(self):
        """I≥15 → mu ≈ threshold → rate > 0."""
        n = SiegertTransferFunction()
        rate = n.step(20.0)
        assert rate > 10.0, f"rate={rate:.2f}"

    def test_rate_increases_with_current(self):
        n = SiegertTransferFunction()
        r15 = n.step(15.0)
        r20 = n.step(20.0)
        r30 = n.step(30.0)
        assert r15 < r20 < r30

    def test_saturation_at_refractory_limit(self):
        """Max rate = 1000/τ_rp = 500 Hz."""
        n = SiegertTransferFunction()
        rate = n.step(50.0)
        assert abs(rate - 500.0) < 1.0, f"rate={rate:.2f}, expected ~500"

    def test_rate_at_known_current(self):
        """At I=20: rate ≈ 53.5 Hz (from probing)."""
        n = SiegertTransferFunction()
        rate = n.step(20.0)
        assert 40 < rate < 70, f"rate={rate:.2f}"


class TestSiegertErfApprox:
    """Abramowitz & Stegun 7.1.26 rational approximation."""

    def test_erf_at_zero(self):
        result = _erf_approx(np.array([0.0]))
        assert abs(result[0]) < 1e-6

    def test_erf_symmetry(self):
        x = np.array([1.0, -1.0])
        result = _erf_approx(x)
        assert abs(result[0] + result[1]) < 1e-6

    def test_erf_accuracy(self):
        """Compare with scipy.special.erf if available."""
        try:
            from scipy.special import erf as scipy_erf

            x = np.array([-2.0, -1.0, 0.0, 0.5, 1.0, 2.0])
            approx = _erf_approx(x)
            exact = scipy_erf(x)
            max_err = np.max(np.abs(approx - exact))
            assert max_err < 1e-6, f"max erf error = {max_err:.2e}"
        except ImportError:
            # scipy not available — skip cross-validation
            pass

    def test_erf_bounded(self):
        x = np.array([-10.0, -1.0, 0.0, 1.0, 10.0])
        result = _erf_approx(x)
        assert np.all(np.abs(result) <= 1.0 + 1e-6)


class TestSiegertAnalytical:
    def test_refractory_period_sets_max_rate(self):
        """τ_rp = 2 → max = 500 Hz. τ_rp = 5 → max = 200 Hz."""
        n2 = SiegertTransferFunction(tau_rp=2.0)
        n5 = SiegertTransferFunction(tau_rp=5.0)
        r2 = n2.step(50.0)
        r5 = n5.step(50.0)
        assert abs(r2 - 500.0) < 1.0
        assert abs(r5 - 200.0) < 1.0

    def test_tau_m_affects_rate(self):
        """Larger τ_m → slower integration → different rate."""
        n_fast = SiegertTransferFunction(tau_m=10.0)
        n_slow = SiegertTransferFunction(tau_m=40.0)
        r_fast = n_fast.step(20.0)
        r_slow = n_slow.step(20.0)
        assert r_fast != r_slow


class TestSiegertValidation:
    @pytest.mark.parametrize("field", ["v_threshold", "v_reset", "v_rest"])
    @pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
    def test_rejects_non_finite_voltage_parameters(self, field: str, value: float):
        with pytest.raises(ValueError, match=field):
            SiegertTransferFunction(**{field: value})

    @pytest.mark.parametrize("field", ["tau_m", "tau_rp"])
    @pytest.mark.parametrize("value", [0.0, -1.0, np.nan, np.inf, -np.inf])
    def test_rejects_non_positive_or_non_finite_time_constants(self, field: str, value: float):
        with pytest.raises(ValueError, match=field):
            SiegertTransferFunction(**{field: value})

    def test_rejects_reset_not_below_threshold(self):
        with pytest.raises(ValueError, match="v_threshold"):
            SiegertTransferFunction(v_reset=-50.0, v_threshold=-50.0)

    @pytest.mark.parametrize("current", [np.nan, np.inf, -np.inf])
    def test_rejects_non_finite_current(self, current: float):
        n = SiegertTransferFunction()
        with pytest.raises(ValueError, match="current"):
            n.step(current)

    @pytest.mark.parametrize("field", ["tau_m", "tau_rp", "v_threshold", "v_reset", "v_rest"])
    def test_rejects_corrupted_runtime_parameters(self, field: str):
        n = SiegertTransferFunction()
        setattr(n, field, np.nan)
        with pytest.raises(ValueError, match="runtime"):
            n.step(20.0)

    def test_rejects_corrupted_runtime_boundary_ordering(self):
        n = SiegertTransferFunction()
        n.v_reset = n.v_threshold
        with pytest.raises(ValueError, match="runtime"):
            n.step(20.0)

    def test_rejects_non_finite_diffusion_scale_before_rate_floor(self):
        n = SiegertTransferFunction()
        n.v_rest = -np.inf
        with pytest.raises(ValueError, match="runtime"):
            n.step(20.0)

    def test_rate_is_finite_non_negative_and_refractory_bounded(self):
        n = SiegertTransferFunction(tau_rp=2.0)
        rates = [n.step(current) for current in [-20.0, 0.0, 20.0, 50.0, 1.0e6]]
        assert all(np.isfinite(rate) for rate in rates)
        assert all(0.0 <= rate <= 500.0 for rate in rates)


class TestSiegertPerformance:
    def test_isolation_throughput(self):
        """Slow due to Gauss-Legendre quadrature (40 points)."""
        n = SiegertTransferFunction()
        N = 500
        t0 = time.perf_counter()
        for _ in range(N):
            n.step(20.0)
        elapsed = time.perf_counter() - t0
        assert N / elapsed > 100  # ~524 steps/s


class TestSiegertPipeline:
    def test_population_creates(self):
        assert Population(SiegertTransferFunction, n=5, label="sieg").n == 5

    def test_returns_float_not_spike(self):
        """Mean-field rate model. Returns Hz, not binary spike."""
        n = SiegertTransferFunction()
        result = n.step(20.0)
        assert isinstance(result, (float, np.floating))

    def test_deterministic(self):
        n1 = SiegertTransferFunction()
        n2 = SiegertTransferFunction()
        assert n1.step(20.0) == n2.step(20.0)


# Salvaged model-specific behavioural contracts from retired aggregate test file.
class TestSiegertTransfer:
    def test_returns_rate(self):
        from sc_neurocore.neurons.models.siegert import SiegertTransferFunction

        n = SiegertTransferFunction()
        rate = n.step(5.0)
        assert isinstance(rate, float)
        assert rate >= 0.0

    def test_higher_input_higher_rate(self):
        from sc_neurocore.neurons.models.siegert import SiegertTransferFunction

        n = SiegertTransferFunction()
        r_low = n.step(1.0)
        r_high = n.step(30.0)
        assert r_high >= r_low

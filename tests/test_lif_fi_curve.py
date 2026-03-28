# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — LIF f-I curve validation against analytical solution

from __future__ import annotations

import math

import pytest

# Normalised LIF: v_rest=0, v_th=1, v_reset=0, R=1, tau=20ms
# Rheobase: I_rh = V_th / (R*tau) = 0.05
DT = 1.0  # dt=1ms — fast, still <5% error for moderate currents


def analytical_lif_fi(
    current: float,
    tau_m: float = 20.0,
    v_threshold: float = 1.0,
    v_reset: float = 0.0,
    v_rest: float = 0.0,
    resistance: float = 1.0,
    refractory_period: int = 0,
    dt: float = DT,
) -> float:
    """Analytical firing rate (spikes/ms) for deterministic LIF.

    v_ss = v_rest + R*I*tau.  ISI = tau * ln((v_ss - v_reset) / (v_ss - v_th)).
    """
    v_ss = v_rest + resistance * current * tau_m
    if v_ss <= v_threshold:
        return 0.0
    isi_ms = tau_m * math.log((v_ss - v_reset) / (v_ss - v_threshold))
    if isi_ms <= 0:
        return 0.0
    t_ref = refractory_period * dt
    return 1.0 / (isi_ms + t_ref)


def simulated_lif_fi(
    current: float,
    duration_ms: float = 500.0,
    tau_m: float = 20.0,
    v_threshold: float = 1.0,
    v_reset: float = 0.0,
    v_rest: float = 0.0,
    resistance: float = 1.0,
    refractory_period: int = 0,
    dt: float = DT,
) -> float:
    """Euler-integrated LIF. Returns spikes/ms."""
    n_steps = int(duration_ms / dt)
    alpha = dt / tau_m
    input_term = resistance * current * dt
    v = v_rest
    spikes = 0
    ref = 0
    for _ in range(n_steps):
        if ref > 0:
            ref -= 1
            continue
        v += -(v - v_rest) * alpha + input_term
        if v >= v_threshold:
            spikes += 1
            v = v_reset
            ref = refractory_period
    return spikes / duration_ms


class TestLIFAnalyticalFICurve:
    """Validate LIF firing rate matches the analytical solution."""

    def test_subthreshold_no_spikes(self):
        for current in [0.01, 0.02, 0.03, 0.04]:
            rate = simulated_lif_fi(current, duration_ms=200.0)
            assert rate == 0.0, f"I={current}: got rate={rate}"

    def test_at_rheobase_no_spikes(self):
        rate = simulated_lif_fi(0.05, duration_ms=200.0)
        assert rate < 0.01, f"At rheobase: rate={rate}"

    def test_suprathreshold_spikes(self):
        for current in [0.06, 0.1, 0.2, 0.5]:
            rate = simulated_lif_fi(current, duration_ms=200.0)
            assert rate > 0, f"I={current}: got rate={rate}"

    def test_fi_curve_monotonic(self):
        currents = [0.06, 0.08, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0]
        rates = [simulated_lif_fi(c) for c in currents]
        for i in range(1, len(rates)):
            assert rates[i] >= rates[i - 1], (
                f"Non-monotonic: f({currents[i - 1]})={rates[i - 1]} > f({currents[i]})={rates[i]}"
            )

    @pytest.mark.parametrize("current", [0.06, 0.08, 0.1, 0.15, 0.2])
    def test_fi_matches_analytical(self, current):
        """Simulated f-I matches analytical within 10% at dt=1ms."""
        f_a = analytical_lif_fi(current)
        f_s = simulated_lif_fi(current, duration_ms=1000.0)
        if f_a == 0:
            assert f_s == 0
            return
        err = abs(f_s - f_a) / f_a
        assert err < 0.10, f"I={current}: analytical={f_a:.6f}, sim={f_s:.6f}, err={err:.2%}"

    def test_refractory_reduces_rate(self):
        r0 = simulated_lif_fi(0.2, duration_ms=500.0)
        r1 = simulated_lif_fi(0.2, duration_ms=500.0, refractory_period=3)
        assert r1 < r0

    def test_different_tau_matches_analytical(self):
        for tau in [10.0, 20.0, 40.0]:
            f_a = analytical_lif_fi(0.15, tau_m=tau)
            f_s = simulated_lif_fi(0.15, duration_ms=1000.0, tau_m=tau)
            err = abs(f_s - f_a) / f_a if f_a > 0 else 0.0
            assert err < 0.10, f"tau={tau}: analytical={f_a:.6f}, sim={f_s:.6f}, err={err:.2%}"

    def test_analytical_formula_known_values(self):
        isi = 20.0 * math.log(2.0)
        f = analytical_lif_fi(0.1)
        assert abs(f - 1.0 / isi) < 1e-10
        assert abs(isi - 13.863) < 0.001

# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

# Source MAT(1) scalar helpers. The caller commits every returned candidate
# from the same pre-step tuple. Units are mV, ms, nA, and megaohms.

from std.math import exp


def _nrlif_finite(value: Float64) -> Bool:
    """Return true only for finite binary64 values."""
    return (
        value == value
        and value <= 1.7976931348623157e308
        and value >= -1.7976931348623157e308
    )


def non_resetting_lif_valid(
    v: Float64,
    theta: Float64,
    refractory: Float64,
    omega: Float64,
    tau_m: Float64,
    tau_theta: Float64,
    alpha: Float64,
    resistance: Float64,
    refractory_period: Float64,
    dt: Float64,
) -> Bool:
    """Validate complete source MAT(1) state and configuration."""
    return (
        _nrlif_finite(v)
        and v >= -200.0
        and v <= 200.0
        and _nrlif_finite(theta)
        and theta >= 0.0
        and theta <= 1.0e9
        and _nrlif_finite(refractory)
        and refractory >= 0.0
        and _nrlif_finite(omega)
        and omega >= -1.0e9
        and omega <= 1.0e9
        and _nrlif_finite(tau_m)
        and tau_m > 0.0
        and _nrlif_finite(tau_theta)
        and tau_theta > 0.0
        and _nrlif_finite(alpha)
        and alpha >= 0.0
        and alpha <= 1.0e9
        and _nrlif_finite(resistance)
        and resistance > 0.0
        and _nrlif_finite(refractory_period)
        and refractory_period >= 0.0
        and refractory <= refractory_period
        and _nrlif_finite(dt)
        and dt > 0.0
    )


def non_resetting_lif_candidate_v(
    v: Float64,
    current: Float64,
    tau_m: Float64,
    resistance: Float64,
    dt: Float64,
) -> Float64:
    """Return the source forward-Euler membrane candidate."""
    return v + dt * (-v + resistance * current) / tau_m


def non_resetting_lif_candidate_theta(
    theta: Float64, tau_theta: Float64, dt: Float64
) -> Float64:
    """Return the exact single threshold-history decay."""
    return theta * exp(-dt / tau_theta)


def non_resetting_lif_candidate_refractory(
    refractory: Float64, dt: Float64
) -> Float64:
    """Return the nonnegative refractory countdown candidate."""
    var value = refractory - dt
    if value < 0.0:
        return 0.0
    return value


def non_resetting_lif_step_spike(
    v: Float64,
    theta: Float64,
    refractory: Float64,
    current: Float64,
    omega: Float64,
    tau_m: Float64,
    tau_theta: Float64,
    alpha: Float64,
    resistance: Float64,
    refractory_period: Float64,
    dt: Float64,
) -> Int:
    """Return one event bit, or -1 for invalid input."""
    if not _nrlif_finite(current) or not non_resetting_lif_valid(
        v,
        theta,
        refractory,
        omega,
        tau_m,
        tau_theta,
        alpha,
        resistance,
        refractory_period,
        dt,
    ):
        return -1
    var nv = non_resetting_lif_candidate_v(v, current, tau_m, resistance, dt)
    var nt = non_resetting_lif_candidate_theta(theta, tau_theta, dt)
    var nr = non_resetting_lif_candidate_refractory(refractory, dt)
    if not (
        _nrlif_finite(nv)
        and nv >= -200.0
        and nv <= 200.0
        and _nrlif_finite(nt)
        and nt >= 0.0
        and nt <= 1.0e9
        and _nrlif_finite(nr)
    ):
        return -1
    if nr == 0.0 and nv >= omega + nt:
        if nt + alpha > 1.0e9:
            return -1
        return 1
    return 0


def non_resetting_lif_next_v(
    v: Float64,
    theta: Float64,
    refractory: Float64,
    current: Float64,
    omega: Float64,
    tau_m: Float64,
    tau_theta: Float64,
    alpha: Float64,
    resistance: Float64,
    refractory_period: Float64,
    dt: Float64,
) -> Float64:
    """Return the non-resetting membrane output, or NaN on invalid input."""
    if (
        non_resetting_lif_step_spike(
            v,
            theta,
            refractory,
            current,
            omega,
            tau_m,
            tau_theta,
            alpha,
            resistance,
            refractory_period,
            dt,
        )
        < 0
    ):
        return 0.0 / 0.0
    return non_resetting_lif_candidate_v(v, current, tau_m, resistance, dt)


def non_resetting_lif_next_theta(
    v: Float64,
    theta: Float64,
    refractory: Float64,
    current: Float64,
    omega: Float64,
    tau_m: Float64,
    tau_theta: Float64,
    alpha: Float64,
    resistance: Float64,
    refractory_period: Float64,
    dt: Float64,
) -> Float64:
    """Return the post-step threshold-history output."""
    var spike = non_resetting_lif_step_spike(
        v,
        theta,
        refractory,
        current,
        omega,
        tau_m,
        tau_theta,
        alpha,
        resistance,
        refractory_period,
        dt,
    )
    if spike < 0:
        return 0.0 / 0.0
    return (
        non_resetting_lif_candidate_theta(theta, tau_theta, dt)
        + Float64(spike) * alpha
    )


def non_resetting_lif_next_refractory(
    v: Float64,
    theta: Float64,
    refractory: Float64,
    current: Float64,
    omega: Float64,
    tau_m: Float64,
    tau_theta: Float64,
    alpha: Float64,
    resistance: Float64,
    refractory_period: Float64,
    dt: Float64,
) -> Float64:
    """Return the post-step absolute-refractory state."""
    var spike = non_resetting_lif_step_spike(
        v,
        theta,
        refractory,
        current,
        omega,
        tau_m,
        tau_theta,
        alpha,
        resistance,
        refractory_period,
        dt,
    )
    if spike < 0:
        return 0.0 / 0.0
    if spike == 1:
        return refractory_period
    return non_resetting_lif_candidate_refractory(refractory, dt)

# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

# Retained SC exact-relaxation helpers. No publication attribution is made.
from std.math import exp


def _sc_nralif_finite(value: Float64) -> Bool:
    """Return true only for finite binary64 values."""
    return (
        value == value
        and value <= 1.7976931348623157e308
        and value >= -1.7976931348623157e308
    )


def sc_non_resetting_adaptive_lif_valid(
    v: Float64,
    theta: Float64,
    v_rest: Float64,
    theta_rest: Float64,
    delta_theta: Float64,
    tau_m: Float64,
    tau_theta: Float64,
    r_m: Float64,
    dt: Float64,
) -> Bool:
    """Validate the complete retained project contract."""
    return (
        _sc_nralif_finite(v)
        and _sc_nralif_finite(theta)
        and _sc_nralif_finite(v_rest)
        and _sc_nralif_finite(theta_rest)
        and _sc_nralif_finite(delta_theta)
        and delta_theta >= 0.0
        and _sc_nralif_finite(tau_m)
        and tau_m > 0.0
        and _sc_nralif_finite(tau_theta)
        and tau_theta > 0.0
        and _sc_nralif_finite(r_m)
        and r_m >= 0.0
        and _sc_nralif_finite(dt)
        and dt > 0.0
    )


def sc_non_resetting_adaptive_lif_candidate_v(
    v: Float64,
    current: Float64,
    v_rest: Float64,
    tau_m: Float64,
    r_m: Float64,
    dt: Float64,
) -> Float64:
    """Return the exact affine membrane relaxation."""
    var decay = exp(-dt / tau_m)
    return decay * v + (1.0 - decay) * (v_rest + r_m * current)


def sc_non_resetting_adaptive_lif_candidate_theta(
    theta: Float64, theta_rest: Float64, tau_theta: Float64, dt: Float64
) -> Float64:
    """Return the exact affine threshold relaxation."""
    var decay = exp(-dt / tau_theta)
    return decay * theta + (1.0 - decay) * theta_rest


def sc_non_resetting_adaptive_lif_step_spike(
    v: Float64,
    theta: Float64,
    current: Float64,
    v_rest: Float64,
    theta_rest: Float64,
    delta_theta: Float64,
    tau_m: Float64,
    tau_theta: Float64,
    r_m: Float64,
    dt: Float64,
) -> Int:
    """Return one event bit, or -1 for invalid input."""
    if not _sc_nralif_finite(
        current
    ) or not sc_non_resetting_adaptive_lif_valid(
        v, theta, v_rest, theta_rest, delta_theta, tau_m, tau_theta, r_m, dt
    ):
        return -1
    var nv = sc_non_resetting_adaptive_lif_candidate_v(
        v, current, v_rest, tau_m, r_m, dt
    )
    var nt = sc_non_resetting_adaptive_lif_candidate_theta(
        theta, theta_rest, tau_theta, dt
    )
    if not (_sc_nralif_finite(nv) and _sc_nralif_finite(nt)):
        return -1
    if nv >= nt:
        if not _sc_nralif_finite(nt + delta_theta):
            return -1
        return 1
    return 0


def sc_non_resetting_adaptive_lif_next_v(
    v: Float64,
    theta: Float64,
    current: Float64,
    v_rest: Float64,
    theta_rest: Float64,
    delta_theta: Float64,
    tau_m: Float64,
    tau_theta: Float64,
    r_m: Float64,
    dt: Float64,
) -> Float64:
    """Return the post-step membrane output, or NaN on invalid input."""
    if (
        sc_non_resetting_adaptive_lif_step_spike(
            v,
            theta,
            current,
            v_rest,
            theta_rest,
            delta_theta,
            tau_m,
            tau_theta,
            r_m,
            dt,
        )
        < 0
    ):
        return 0.0 / 0.0
    return sc_non_resetting_adaptive_lif_candidate_v(
        v, current, v_rest, tau_m, r_m, dt
    )


def sc_non_resetting_adaptive_lif_next_theta(
    v: Float64,
    theta: Float64,
    current: Float64,
    v_rest: Float64,
    theta_rest: Float64,
    delta_theta: Float64,
    tau_m: Float64,
    tau_theta: Float64,
    r_m: Float64,
    dt: Float64,
) -> Float64:
    """Return the post-step adaptive-threshold output."""
    var spike = sc_non_resetting_adaptive_lif_step_spike(
        v,
        theta,
        current,
        v_rest,
        theta_rest,
        delta_theta,
        tau_m,
        tau_theta,
        r_m,
        dt,
    )
    if spike < 0:
        return 0.0 / 0.0
    return (
        sc_non_resetting_adaptive_lif_candidate_theta(
            theta, theta_rest, tau_theta, dt
        )
        + Float64(spike) * delta_theta
    )

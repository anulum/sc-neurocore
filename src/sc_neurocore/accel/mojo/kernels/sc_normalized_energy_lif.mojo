# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo retained normalized EnergyLIF kernel

from std.math import expm1


def _sc_energy_finite(x: Float64) -> Bool:
    return x == x and x <= 1.7976931348623157e308 and x >= -1.7976931348623157e308


def sc_normalized_energy_lif_valid(v: Float64, e: Float64, vrest: Float64, vreset: Float64, vth: Float64, tau_m: Float64, tau_e: Float64, alpha: Float64, epsilon0: Float64, resistance: Float64, dt: Float64) -> Bool:
    """Validate the complete current state and configured resting state."""
    return (_sc_energy_finite(v) and _sc_energy_finite(e) and
            _sc_energy_finite(vrest) and _sc_energy_finite(vreset) and
            _sc_energy_finite(vth) and _sc_energy_finite(tau_m) and
            _sc_energy_finite(tau_e) and _sc_energy_finite(alpha) and
            _sc_energy_finite(epsilon0) and _sc_energy_finite(resistance) and
            _sc_energy_finite(dt) and v >= -200.0 and v <= 100.0 and
            vrest >= -200.0 and vrest <= 100.0 and vreset >= -200.0 and
            vreset <= 100.0 and e >= 0.0 and e <= epsilon0 and epsilon0 >= 0.0 and
            tau_m > 0.0 and tau_e > 0.0 and alpha >= 0.0 and resistance > 0.0 and
            dt > 0.0 and dt <= tau_m and dt <= tau_e and vth > vrest and vth > vreset)


def sc_normalized_energy_lif_next_epsilon(e: Float64, epsilon0: Float64, tau_e: Float64, dt: Float64) -> Float64:
    """Return the retained exact energy relaxation candidate."""
    return epsilon0 + (e - epsilon0) * (1.0 + expm1(-dt / tau_e))


def sc_normalized_energy_lif_next_v(v: Float64, e: Float64, vrest: Float64, tau_m: Float64, tau_e: Float64, epsilon0: Float64, resistance: Float64, dt: Float64, current: Float64) -> Float64:
    """Return the retained constant-current voltage candidate."""
    # Valid dt/tau lies in (0, 1]; expm1 retains precision at the -1 endpoint.
    var md = 1.0 + expm1(-dt / tau_m)
    var de = e - epsilon0
    var steady = epsilon0 * tau_m * (1.0 - md)
    var rate = 1.0 / tau_m - 1.0 / tau_e
    var transient = de * md * dt
    if rate >= 1.0e-12 or rate <= -1.0e-12:
        transient = de * md * expm1(rate * dt) / rate
    return vrest + (v - vrest) * md + (resistance * current / tau_m) * (steady + transient)


def sc_normalized_energy_lif_step_spike(v: Float64, e: Float64, vrest: Float64, vreset: Float64, vth: Float64, tau_m: Float64, tau_e: Float64, alpha: Float64, epsilon0: Float64, resistance: Float64, dt: Float64, current: Float64) -> Int:
    """Return a retained event, or minus one before any unsafe state commit."""
    if not sc_normalized_energy_lif_valid(v, e, vrest, vreset, vth, tau_m, tau_e, alpha, epsilon0, resistance, dt) or not _sc_energy_finite(current):
        return -1
    var nv = sc_normalized_energy_lif_next_v(v, e, vrest, tau_m, tau_e, epsilon0, resistance, dt, current)
    var ne = sc_normalized_energy_lif_next_epsilon(e, epsilon0, tau_e, dt)
    if not (_sc_energy_finite(nv) and nv >= -200.0 and nv <= 100.0 and _sc_energy_finite(ne) and ne >= 0.0 and ne <= epsilon0):
        return -1
    if nv >= vth and ne > 0.1:
        return 1
    return 0

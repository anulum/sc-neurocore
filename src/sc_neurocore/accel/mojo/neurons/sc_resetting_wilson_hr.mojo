# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo retained resetting Wilson-HR project recurrence
#
# Build: mojo build --emit shared-lib -o libsc_resetting_wilson_hr.so
#        sc_resetting_wilson_hr.mojo
#
# ABI contract: the caller owns n+2 Float64 slots. The kernel writes the
# post-step voltage trace, then final voltage and recovery state. It returns -1
# on invalid input or arithmetic, and the caller must leave model state intact.

from std.math import isfinite
from std.memory import UnsafePointer


@export
def sc_resetting_wilson_hr_simulate_c(
    v0: Float64,
    r0: Float64,
    tau_r: Float64,
    v_peak: Float64,
    dt: Float64,
    n_steps: Int,
    current: Float64,
    trace_addr: Int,
) -> Int64:
    """Run the retained unit-capacitance RK4 recurrence with hard reset."""
    if (
        n_steps < 0
        or trace_addr == 0
        or not isfinite(v0)
        or not isfinite(r0)
        or not isfinite(tau_r)
        or not isfinite(v_peak)
        or not isfinite(dt)
        or not isfinite(current)
        or tau_r <= 0.0
        or dt <= 0.0
    ):
        return -1
    var trace = UnsafePointer[Float64, MutAnyOrigin](unsafe_from_address=trace_addr)
    var v = v0
    var r = r0
    var events: Int64 = 0
    for index in range(n_steps):
        var dv1 = _dv(v, r, current)
        var dr1 = _dr(v, r, tau_r)
        var half_step = 0.5 * dt
        var dv2 = _dv(v + half_step * dv1, r + half_step * dr1, current)
        var dr2 = _dr(v + half_step * dv1, r + half_step * dr1, tau_r)
        var dv3 = _dv(v + half_step * dv2, r + half_step * dr2, current)
        var dr3 = _dr(v + half_step * dv2, r + half_step * dr2, tau_r)
        var dv4 = _dv(v + dt * dv3, r + dt * dr3, current)
        var dr4 = _dr(v + dt * dv3, r + dt * dr3, tau_r)
        if (
            not isfinite(dv1)
            or not isfinite(dr1)
            or not isfinite(dv2)
            or not isfinite(dr2)
            or not isfinite(dv3)
            or not isfinite(dr3)
            or not isfinite(dv4)
            or not isfinite(dr4)
        ):
            return -1
        var sum_v = dv1 + 2.0 * dv2 + 2.0 * dv3 + dv4
        var sum_r = dr1 + 2.0 * dr2 + 2.0 * dr3 + dr4
        var next_v = v + dt * sum_v / 6.0
        var next_r = r + dt * sum_r / 6.0
        if not isfinite(next_v) or not isfinite(next_r):
            return -1
        if next_v >= v_peak:
            next_v = -0.7
            events += 1
        v = next_v
        r = next_r
        trace[index] = v
    trace[n_steps] = v
    trace[n_steps + 1] = r
    return events


@always_inline
def _dv(v: Float64, r: Float64, current: Float64) -> Float64:
    var quadratic = 32.63 * v * v
    var linear = 47.71 * v
    var polynomial = -(17.81 + linear + quadratic) * (v - 0.55)
    var recovery_current = -26.0 * r * (v + 0.92)
    return polynomial + recovery_current + current


@always_inline
def _dr(v: Float64, r: Float64, tau_r: Float64) -> Float64:
    var numerator = -r + 1.35 * v + 1.03
    return numerator / tau_r

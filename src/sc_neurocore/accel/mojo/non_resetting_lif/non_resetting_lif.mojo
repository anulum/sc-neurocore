# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Build: mojo build --emit shared-lib -o libnon_resetting_lif.so non_resetting_lif.mojo

from std.math import exp
from std.memory import UnsafePointer


def _finite(value: Float64) -> Bool:
    """Return true only for finite binary64 values."""
    return (
        value == value
        and value <= 1.7976931348623157e308
        and value >= -1.7976931348623157e308
    )


@export
def non_resetting_lif_simulate_c(
    steps: Int,
    v_init: Float64,
    theta_init: Float64,
    refractory_init: Float64,
    omega: Float64,
    tau_m: Float64,
    tau_theta: Float64,
    alpha: Float64,
    resistance: Float64,
    refractory_period: Float64,
    dt: Float64,
    currents_addr: Int,
    voltages_addr: Int,
    theta_addr: Int,
    refractory_addr: Int,
    events_addr: Int,
    v_final_addr: Int,
    theta_final_addr: Int,
    refractory_final_addr: Int,
) -> Int:
    """Run a complete configured source MAT(1) batch."""
    if steps < 0:
        return 1
    var currents = UnsafePointer[Float64, MutAnyOrigin](
        unsafe_from_address=currents_addr
    )
    var voltages = UnsafePointer[Float64, MutAnyOrigin](
        unsafe_from_address=voltages_addr
    )
    var thresholds = UnsafePointer[Float64, MutAnyOrigin](
        unsafe_from_address=theta_addr
    )
    var refractory_trace = UnsafePointer[Float64, MutAnyOrigin](
        unsafe_from_address=refractory_addr
    )
    var events = UnsafePointer[Int64, MutAnyOrigin](
        unsafe_from_address=events_addr
    )
    var v_final = UnsafePointer[Float64, MutAnyOrigin](
        unsafe_from_address=v_final_addr
    )
    var theta_final = UnsafePointer[Float64, MutAnyOrigin](
        unsafe_from_address=theta_final_addr
    )
    var refractory_final = UnsafePointer[Float64, MutAnyOrigin](
        unsafe_from_address=refractory_final_addr
    )
    var v = v_init
    var theta = theta_init
    var refractory = refractory_init
    for index in range(steps):
        var current = currents[index]
        if not (
            _finite(v)
            and v >= -200.0
            and v <= 200.0
            and _finite(theta)
            and theta >= 0.0
            and theta <= 1.0e9
            and _finite(refractory)
            and refractory >= 0.0
            and _finite(current)
            and _finite(omega)
            and _finite(tau_m)
            and tau_m > 0.0
            and _finite(tau_theta)
            and tau_theta > 0.0
            and _finite(alpha)
            and alpha >= 0.0
            and _finite(resistance)
            and resistance > 0.0
            and _finite(refractory_period)
            and refractory_period >= 0.0
            and refractory <= refractory_period
            and _finite(dt)
            and dt > 0.0
        ):
            return 2
        var nv = v + dt * (-v + resistance * current) / tau_m
        var nt = theta * exp(-dt / tau_theta)
        var nr = refractory - dt
        if nr < 0.0:
            nr = 0.0
        if not (
            _finite(nv)
            and nv >= -200.0
            and nv <= 200.0
            and _finite(nt)
            and nt >= 0.0
            and nt <= 1.0e9
        ):
            return 2
        var spike = Int(nr == 0.0 and nv >= omega + nt)
        if spike == 1:
            nt += alpha
            nr = refractory_period
        if not _finite(nt) or nt > 1.0e9:
            return 2
        v = nv
        theta = nt
        refractory = nr
        voltages[index] = v
        thresholds[index] = theta
        refractory_trace[index] = refractory
        events[index] = Int64(spike)
    v_final[0] = v
    theta_final[0] = theta
    refractory_final[0] = refractory
    return 0

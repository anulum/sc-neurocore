# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Build: mojo build --emit shared-lib -o libsc_non_resetting_adaptive_lif.so sc_non_resetting_adaptive_lif.mojo

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
def sc_non_resetting_adaptive_lif_simulate_c(
    steps: Int,
    v_init: Float64,
    theta_init: Float64,
    v_rest: Float64,
    theta_rest: Float64,
    delta_theta: Float64,
    tau_m: Float64,
    tau_theta: Float64,
    r_m: Float64,
    dt: Float64,
    currents_addr: Int,
    voltages_addr: Int,
    theta_addr: Int,
    events_addr: Int,
    v_final_addr: Int,
    theta_final_addr: Int,
) -> Int:
    """Run a complete retained-project exact-relaxation batch."""
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
    var events = UnsafePointer[Int64, MutAnyOrigin](
        unsafe_from_address=events_addr
    )
    var v_final = UnsafePointer[Float64, MutAnyOrigin](
        unsafe_from_address=v_final_addr
    )
    var theta_final = UnsafePointer[Float64, MutAnyOrigin](
        unsafe_from_address=theta_final_addr
    )
    var v = v_init
    var theta = theta_init
    for index in range(steps):
        var current = currents[index]
        if not (
            _finite(v)
            and _finite(theta)
            and _finite(v_rest)
            and _finite(theta_rest)
            and _finite(delta_theta)
            and delta_theta >= 0.0
            and _finite(tau_m)
            and tau_m > 0.0
            and _finite(tau_theta)
            and tau_theta > 0.0
            and _finite(r_m)
            and r_m >= 0.0
            and _finite(dt)
            and dt > 0.0
            and _finite(current)
        ):
            return 2
        var steady = v_rest + r_m * current
        var dv = exp(-dt / tau_m)
        var dtheta = exp(-dt / tau_theta)
        var nv = dv * v + (1.0 - dv) * steady
        var nt = dtheta * theta + (1.0 - dtheta) * theta_rest
        if not (_finite(steady) and _finite(nv) and _finite(nt)):
            return 2
        var spike = Int(nv >= nt)
        if spike == 1:
            nt += delta_theta
        if not _finite(nt):
            return 2
        v = nv
        theta = nt
        voltages[index] = v
        thresholds[index] = theta
        events[index] = Int64(spike)
    v_final[0] = v
    theta_final[0] = theta
    return 0

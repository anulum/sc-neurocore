# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo C ABI for source MAT* batches
# Build: mojo build --emit shared-lib -o libmat.so mat.mojo

from std.math import exp
from std.memory import UnsafePointer


def _finite(value: Float64) -> Bool:
    """Return true only for finite binary64 values."""
    return value == value and value <= 1.7976931348623157e308 and value >= -1.7976931348623157e308


@export
def mat_simulate_c(
    steps: Int,
    v_init: Float64,
    theta1_init: Float64,
    theta2_init: Float64,
    refractory_init: Float64,
    omega: Float64,
    tau_m: Float64,
    tau_1: Float64,
    tau_2: Float64,
    alpha_1: Float64,
    alpha_2: Float64,
    resistance: Float64,
    refractory_period: Float64,
    dt: Float64,
    currents_addr: Int,
    voltages_addr: Int,
    theta1_addr: Int,
    theta2_addr: Int,
    refractory_addr: Int,
    events_addr: Int,
    v_final_addr: Int,
    theta1_final_addr: Int,
    theta2_final_addr: Int,
    refractory_final_addr: Int,
) -> Int:
    """Run a complete configured non-resetting MAT* batch."""
    if steps < 0:
        return 1
    var currents = UnsafePointer[Float64, MutAnyOrigin](unsafe_from_address=currents_addr)
    var voltages = UnsafePointer[Float64, MutAnyOrigin](unsafe_from_address=voltages_addr)
    var theta1_trace = UnsafePointer[Float64, MutAnyOrigin](unsafe_from_address=theta1_addr)
    var theta2_trace = UnsafePointer[Float64, MutAnyOrigin](unsafe_from_address=theta2_addr)
    var refractory_trace = UnsafePointer[Float64, MutAnyOrigin](unsafe_from_address=refractory_addr)
    var events = UnsafePointer[Int64, MutAnyOrigin](unsafe_from_address=events_addr)
    var v_final = UnsafePointer[Float64, MutAnyOrigin](unsafe_from_address=v_final_addr)
    var theta1_final = UnsafePointer[Float64, MutAnyOrigin](unsafe_from_address=theta1_final_addr)
    var theta2_final = UnsafePointer[Float64, MutAnyOrigin](unsafe_from_address=theta2_final_addr)
    var refractory_final = UnsafePointer[Float64, MutAnyOrigin](unsafe_from_address=refractory_final_addr)
    var v = v_init
    var theta1 = theta1_init
    var theta2 = theta2_init
    var refractory = refractory_init
    for index in range(steps):
        var current = currents[index]
        if not (
            _finite(v) and v >= -200.0 and v <= 200.0
            and _finite(theta1) and theta1 >= 0.0 and theta1 <= 1.0e9
            and _finite(theta2) and theta2 >= 0.0 and theta2 <= 1.0e9
            and _finite(refractory) and refractory >= 0.0
            and _finite(current) and _finite(omega)
            and _finite(tau_m) and tau_m > 0.0
            and _finite(tau_1) and tau_1 > 0.0
            and _finite(tau_2) and tau_2 > 0.0
            and _finite(alpha_1) and alpha_1 >= 0.0
            and _finite(alpha_2) and alpha_2 >= 0.0
            and _finite(resistance) and resistance > 0.0
            and _finite(refractory_period) and refractory_period >= 0.0
            and refractory <= refractory_period
            and _finite(dt) and dt > 0.0
        ):
            return 2
        var next_v = v + dt * (-v + resistance * current) / tau_m
        var next_theta1 = theta1 * exp(-dt / tau_1)
        var next_theta2 = theta2 * exp(-dt / tau_2)
        var next_refractory = refractory - dt
        if next_refractory < 0.0:
            next_refractory = 0.0
        if not (
            _finite(next_v) and next_v >= -200.0 and next_v <= 200.0
            and _finite(next_theta1) and next_theta1 >= 0.0 and next_theta1 <= 1.0e9
            and _finite(next_theta2) and next_theta2 >= 0.0 and next_theta2 <= 1.0e9
        ):
            return 2
        var spike = Int(next_refractory == 0.0 and next_v >= omega + next_theta1 + next_theta2)
        if spike == 1:
            next_theta1 += alpha_1
            next_theta2 += alpha_2
            next_refractory = refractory_period
            if next_theta1 > 1.0e9 or next_theta2 > 1.0e9:
                return 2
        v = next_v
        theta1 = next_theta1
        theta2 = next_theta2
        refractory = next_refractory
        voltages[index] = v
        theta1_trace[index] = theta1
        theta2_trace[index] = theta2
        refractory_trace[index] = refractory
        events[index] = Int64(spike)
    v_final[0] = v
    theta1_final[0] = theta1
    theta2_final[0] = theta2
    refractory_final[0] = refractory
    return 0

# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo C ABI for SC resetting-MAT batches
# Build: mojo build --emit shared-lib -o libsc_resetting_mat.so sc_resetting_mat.mojo

from std.memory import UnsafePointer


def _finite(value: Float64) -> Bool:
    """Return true only for finite binary64 values."""
    return value == value and value <= 1.7976931348623157e308 and value >= -1.7976931348623157e308


def _rk4_linear(value: Float64, equilibrium: Float64, tau: Float64, dt: Float64) -> Float64:
    """Advance one affine linear state with classical RK4."""
    var k1 = -(value - equilibrium) / tau
    var k2 = -(value + 0.5 * dt * k1 - equilibrium) / tau
    var k3 = -(value + 0.5 * dt * k2 - equilibrium) / tau
    var k4 = -(value + dt * k3 - equilibrium) / tau
    return value + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


@export
def sc_resetting_mat_simulate_c(
    steps: Int,
    v_init: Float64,
    theta1_init: Float64,
    theta2_init: Float64,
    v_rest: Float64,
    v_reset: Float64,
    v_threshold_base: Float64,
    tau_m: Float64,
    tau_1: Float64,
    tau_2: Float64,
    h1: Float64,
    h2: Float64,
    resistance: Float64,
    dt: Float64,
    currents_addr: Int,
    voltages_addr: Int,
    theta1_addr: Int,
    theta2_addr: Int,
    events_addr: Int,
    v_final_addr: Int,
    theta1_final_addr: Int,
    theta2_final_addr: Int,
) -> Int:
    """Run a complete configured SC candidate-first RK4/reset batch."""
    if steps < 0:
        return 1
    var currents = UnsafePointer[Float64, MutAnyOrigin](unsafe_from_address=currents_addr)
    var voltages = UnsafePointer[Float64, MutAnyOrigin](unsafe_from_address=voltages_addr)
    var theta1_trace = UnsafePointer[Float64, MutAnyOrigin](unsafe_from_address=theta1_addr)
    var theta2_trace = UnsafePointer[Float64, MutAnyOrigin](unsafe_from_address=theta2_addr)
    var events = UnsafePointer[Int64, MutAnyOrigin](unsafe_from_address=events_addr)
    var v_final = UnsafePointer[Float64, MutAnyOrigin](unsafe_from_address=v_final_addr)
    var theta1_final = UnsafePointer[Float64, MutAnyOrigin](unsafe_from_address=theta1_final_addr)
    var theta2_final = UnsafePointer[Float64, MutAnyOrigin](unsafe_from_address=theta2_final_addr)
    var v = v_init
    var theta1 = theta1_init
    var theta2 = theta2_init
    for index in range(steps):
        var current = currents[index]
        if not (
            _finite(v) and v >= -200.0 and v <= 100.0
            and _finite(theta1) and theta1 >= 0.0 and theta1 <= 1.0e9
            and _finite(theta2) and theta2 >= 0.0 and theta2 <= 1.0e9
            and _finite(current) and _finite(v_rest)
            and _finite(v_reset) and v_reset >= -200.0 and v_reset <= 100.0
            and _finite(v_threshold_base)
            and _finite(tau_m) and tau_m > 0.0
            and _finite(tau_1) and tau_1 > 0.0
            and _finite(tau_2) and tau_2 > 0.0
            and _finite(h1) and h1 >= 0.0
            and _finite(h2) and h2 >= 0.0
            and _finite(resistance) and resistance > 0.0
            and _finite(dt) and dt > 0.0
        ):
            return 2
        var next_v = _rk4_linear(v, v_rest + resistance * current, tau_m, dt)
        var next_theta1 = _rk4_linear(theta1, 0.0, tau_1, dt)
        var next_theta2 = _rk4_linear(theta2, 0.0, tau_2, dt)
        if not (
            _finite(next_v) and next_v >= -200.0 and next_v <= 100.0
            and _finite(next_theta1) and next_theta1 >= 0.0 and next_theta1 <= 1.0e9
            and _finite(next_theta2) and next_theta2 >= 0.0 and next_theta2 <= 1.0e9
        ):
            return 2
        var spike = Int(next_v >= v_threshold_base + next_theta1 + next_theta2)
        if spike == 1:
            v = v_reset
            theta1 = next_theta1 + h1
            theta2 = next_theta2 + h2
            if theta1 > 1.0e9 or theta2 > 1.0e9:
                return 2
        else:
            v = next_v
            theta1 = next_theta1
            theta2 = next_theta2
        voltages[index] = v
        theta1_trace[index] = theta1
        theta2_trace[index] = theta2
        events[index] = Int64(spike)
    v_final[0] = v
    theta1_final[0] = theta1
    theta2_final[0] = theta2
    return 0

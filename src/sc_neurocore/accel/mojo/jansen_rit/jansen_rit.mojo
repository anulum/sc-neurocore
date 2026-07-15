# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo batch mirror for Jansen–Rit 1995

from std.math import exp, isfinite
from std.memory import UnsafePointer


@always_inline
def sigmoid(voltage: Float64, e0: Float64, v0: Float64, slope: Float64) -> Float64:
    var exponent = slope * (v0 - voltage)
    if exponent >= 0.0:
        var exp_neg = exp(-exponent)
        return 2.0 * e0 * exp_neg / (1.0 + exp_neg)
    return 2.0 * e0 / (1.0 + exp(exponent))


@always_inline
def valid_configuration(
    y0: Float64,
    y3: Float64,
    y1: Float64,
    y4: Float64,
    y2: Float64,
    y5: Float64,
    a_exc: Float64,
    b_exc: Float64,
    a_rate: Float64,
    b_rate: Float64,
    c: Float64,
    e0: Float64,
    v0: Float64,
    slope: Float64,
    dt: Float64,
) -> Bool:
    return (
        isfinite(y0)
        and isfinite(y3)
        and isfinite(y1)
        and isfinite(y4)
        and isfinite(y2)
        and isfinite(y5)
        and isfinite(a_exc)
        and a_exc > 0.0
        and isfinite(b_exc)
        and b_exc > 0.0
        and isfinite(a_rate)
        and a_rate > 0.0
        and isfinite(b_rate)
        and b_rate > 0.0
        and isfinite(c)
        and c >= 0.0
        and isfinite(e0)
        and e0 > 0.0
        and isfinite(v0)
        and isfinite(slope)
        and slope > 0.0
        and isfinite(dt)
        and dt > 0.0
    )


@export
def jansen_rit_simulate_c(
    n: Int,
    y0_init: Float64,
    y3_init: Float64,
    y1_init: Float64,
    y4_init: Float64,
    y2_init: Float64,
    y5_init: Float64,
    a_exc: Float64,
    b_exc: Float64,
    a_rate: Float64,
    b_rate: Float64,
    c: Float64,
    e0: Float64,
    v0: Float64,
    slope: Float64,
    dt: Float64,
    p_ext_addr: Int,
    y0_out_addr: Int,
    y3_out_addr: Int,
    y1_out_addr: Int,
    y4_out_addr: Int,
    y2_out_addr: Int,
    y5_out_addr: Int,
    eeg_out_addr: Int,
    y0_final_addr: Int,
    y3_final_addr: Int,
    y1_final_addr: Int,
    y4_final_addr: Int,
    y2_final_addr: Int,
    y5_final_addr: Int,
) -> Int:
    if n < 0:
        return 1
    if not valid_configuration(
        y0_init,
        y3_init,
        y1_init,
        y4_init,
        y2_init,
        y5_init,
        a_exc,
        b_exc,
        a_rate,
        b_rate,
        c,
        e0,
        v0,
        slope,
        dt,
    ):
        return 2

    var p_ext = UnsafePointer[Float64, MutAnyOrigin](unsafe_from_address=p_ext_addr)
    var y0_out = UnsafePointer[Float64, MutAnyOrigin](unsafe_from_address=y0_out_addr)
    var y3_out = UnsafePointer[Float64, MutAnyOrigin](unsafe_from_address=y3_out_addr)
    var y1_out = UnsafePointer[Float64, MutAnyOrigin](unsafe_from_address=y1_out_addr)
    var y4_out = UnsafePointer[Float64, MutAnyOrigin](unsafe_from_address=y4_out_addr)
    var y2_out = UnsafePointer[Float64, MutAnyOrigin](unsafe_from_address=y2_out_addr)
    var y5_out = UnsafePointer[Float64, MutAnyOrigin](unsafe_from_address=y5_out_addr)
    var eeg_out = UnsafePointer[Float64, MutAnyOrigin](unsafe_from_address=eeg_out_addr)
    var y0_final = UnsafePointer[Float64, MutAnyOrigin](unsafe_from_address=y0_final_addr)
    var y3_final = UnsafePointer[Float64, MutAnyOrigin](unsafe_from_address=y3_final_addr)
    var y1_final = UnsafePointer[Float64, MutAnyOrigin](unsafe_from_address=y1_final_addr)
    var y4_final = UnsafePointer[Float64, MutAnyOrigin](unsafe_from_address=y4_final_addr)
    var y2_final = UnsafePointer[Float64, MutAnyOrigin](unsafe_from_address=y2_final_addr)
    var y5_final = UnsafePointer[Float64, MutAnyOrigin](unsafe_from_address=y5_final_addr)

    for index in range(n):
        if not isfinite(p_ext[index]):
            return 3

    var y0 = y0_init
    var y3 = y3_init
    var y1 = y1_init
    var y4 = y4_init
    var y2 = y2_init
    var y5 = y5_init
    var c2 = 0.8 * c
    var c3 = 0.25 * c
    var c4 = 0.25 * c
    for index in range(n):
        var s_pyramidal = sigmoid(y1 - y2, e0, v0, slope)
        var s_excitatory = sigmoid(c * y0, e0, v0, slope)
        var s_inhibitory = sigmoid(c3 * y0, e0, v0, slope)
        var next_y0 = y0 + dt * y3
        var next_y3 = y3 + dt * (
            a_exc * a_rate * s_pyramidal
            - 2.0 * a_rate * y3
            - a_rate * a_rate * y0
        )
        var next_y1 = y1 + dt * y4
        var next_y4 = y4 + dt * (
            a_exc * a_rate * (p_ext[index] + c2 * s_excitatory)
            - 2.0 * a_rate * y4
            - a_rate * a_rate * y1
        )
        var next_y2 = y2 + dt * y5
        var next_y5 = y5 + dt * (
            b_exc * b_rate * c4 * s_inhibitory
            - 2.0 * b_rate * y5
            - b_rate * b_rate * y2
        )
        if not (
            isfinite(next_y0)
            and isfinite(next_y3)
            and isfinite(next_y1)
            and isfinite(next_y4)
            and isfinite(next_y2)
            and isfinite(next_y5)
        ):
            return 4
        y0, y3, y1 = next_y0, next_y3, next_y1
        y4, y2, y5 = next_y4, next_y2, next_y5
        y0_out[index], y3_out[index] = y0, y3
        y1_out[index], y4_out[index] = y1, y4
        y2_out[index], y5_out[index] = y2, y5
        eeg_out[index] = y1 - y2

    y0_final[0], y3_final[0] = y0, y3
    y1_final[0], y4_final[0] = y1, y4
    y2_final[0], y5_final[0] = y2, y5
    return 0

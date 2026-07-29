# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo C ABI for the Amari 1977 periodic neural field

from std.math import exp, isfinite
from std.memory import UnsafePointer


@always_inline
def kernel_value(
    distance: Float64,
    a_exc: Float64,
    a_width: Float64,
    b_inh: Float64,
    b_width: Float64,
) -> Float64:
    return a_exc * exp(-a_width * distance) - b_inh * exp(-b_width * distance)


@export
def amari_field_simulate_c(
    steps: Int,
    n: Int,
    tau: Float64,
    a_exc: Float64,
    a_width: Float64,
    b_inh: Float64,
    b_width: Float64,
    dx: Float64,
    dt: Float64,
    u_init_addr: Int,
    currents_addr: Int,
    states_out_addr: Int,
    rates_out_addr: Int,
    final_out_addr: Int,
) -> Int:
    """Advance a flattened steps-by-sites batch; return zero on full success."""
    if steps < 0 or n < 2:
        return 1
    if not (
        isfinite(tau) and tau > 0.0 and isfinite(a_exc) and a_exc >= 0.0
        and isfinite(a_width) and a_width > 0.0 and isfinite(b_inh) and b_inh >= 0.0
        and isfinite(b_width) and b_width > 0.0 and isfinite(dx) and dx > 0.0
        and isfinite(dt) and dt > 0.0
    ):
        return 2
    var u_init = UnsafePointer[Float64, MutAnyOrigin](unsafe_from_address=u_init_addr)
    var currents = UnsafePointer[Float64, MutAnyOrigin](unsafe_from_address=currents_addr)
    var states_out = UnsafePointer[Float64, MutAnyOrigin](unsafe_from_address=states_out_addr)
    var rates_out = UnsafePointer[Float64, MutAnyOrigin](unsafe_from_address=rates_out_addr)
    var final_out = UnsafePointer[Float64, MutAnyOrigin](unsafe_from_address=final_out_addr)
    for i in range(n):
        if not isfinite(u_init[i]):
            return 3
    for index in range(steps * n):
        if not isfinite(currents[index]):
            return 3
    var far_distance = Float64(n // 2) * dx
    if kernel_value(0.0, a_exc, a_width, b_inh, b_width) <= 0.0:
        return 4
    if kernel_value(far_distance, a_exc, a_width, b_inh, b_width) >= 0.0:
        return 4
    # Caller-owned state output doubles as scratch space. No pointer ownership
    # crosses the ABI, and nonzero status invalidates all output buffers.
    for i in range(n):
        final_out[i] = u_init[i]
    for step in range(steps):
        var active = 0
        for i in range(n):
            var convolution = 0.0
            for j in range(n):
                if final_out[j] > 0.0:
                    var offset = i - j
                    if offset < 0:
                        offset += n
                    var wrapped = offset
                    if n - offset < wrapped:
                        wrapped = n - offset
                    convolution += kernel_value(
                        Float64(wrapped) * dx, a_exc, a_width, b_inh, b_width
                    )
            var candidate = final_out[i] + (
                -final_out[i] + convolution * dx + currents[step * n + i]
            ) * (dt / tau)
            if not isfinite(candidate):
                return 5
            states_out[step * n + i] = candidate
        for i in range(n):
            final_out[i] = states_out[step * n + i]
            if final_out[i] > 0.0:
                active += 1
        rates_out[step] = Float64(active) / Float64(n)
    return 0

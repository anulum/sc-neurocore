# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Atomic Mojo batch mirror for the MPR mean field

from std.math import isfinite
from std.memory import UnsafePointer


@always_inline
def valid_configuration(
    r: Float64,
    v: Float64,
    tau: Float64,
    delta: Float64,
    eta_bar: Float64,
    coupling: Float64,
    dt: Float64,
) -> Bool:
    return (
        isfinite(r)
        and r >= 0.0
        and isfinite(v)
        and isfinite(tau)
        and tau > 0.0
        and isfinite(delta)
        and delta >= 0.0
        and isfinite(eta_bar)
        and isfinite(coupling)
        and isfinite(dt)
        and dt > 0.0
    )


@always_inline
def _ranges_overlap(
    a_addr: Int,
    a_elements: Int,
    b_addr: Int,
    b_elements: Int,
) -> Bool:
    var a_bytes = a_elements * 8
    var b_bytes = b_elements * 8
    if a_addr <= b_addr:
        return b_addr - a_addr < a_bytes
    return a_addr - b_addr < b_bytes


@always_inline
def _active_regions_overlap(
    steps: Int,
    ext_addr: Int,
    r_out_addr: Int,
    v_out_addr: Int,
    r_final_addr: Int,
    v_final_addr: Int,
) -> Bool:
    if _ranges_overlap(r_final_addr, 1, v_final_addr, 1):
        return True
    if steps == 0:
        return False
    return (
        _ranges_overlap(ext_addr, steps, r_out_addr, steps)
        or _ranges_overlap(ext_addr, steps, v_out_addr, steps)
        or _ranges_overlap(ext_addr, steps, r_final_addr, 1)
        or _ranges_overlap(ext_addr, steps, v_final_addr, 1)
        or _ranges_overlap(r_out_addr, steps, v_out_addr, steps)
        or _ranges_overlap(r_out_addr, steps, r_final_addr, 1)
        or _ranges_overlap(r_out_addr, steps, v_final_addr, 1)
        or _ranges_overlap(v_out_addr, steps, r_final_addr, 1)
        or _ranges_overlap(v_out_addr, steps, v_final_addr, 1)
    )


def _run_ermentrout_kopell_pop(
    n: Int32,
    r_init: Float64,
    v_init: Float64,
    tau: Float64,
    delta: Float64,
    eta_bar: Float64,
    coupling: Float64,
    dt: Float64,
    ext_addr: Int,
    r_out_addr: Int,
    v_out_addr: Int,
    r_final_addr: Int,
    v_final_addr: Int,
    write_output: Bool,
) -> Int32:
    if n < 0 or r_final_addr == 0 or v_final_addr == 0:
        return 1
    var steps = Int(n)
    if steps > 0 and (ext_addr == 0 or r_out_addr == 0 or v_out_addr == 0):
        return 1
    if _active_regions_overlap(
        steps,
        ext_addr,
        r_out_addr,
        v_out_addr,
        r_final_addr,
        v_final_addr,
    ):
        return 1
    if not valid_configuration(r_init, v_init, tau, delta, eta_bar, coupling, dt):
        return 2

    var r_final = UnsafePointer[Float64, MutAnyOrigin](
        unsafe_from_address=r_final_addr
    )
    var v_final = UnsafePointer[Float64, MutAnyOrigin](
        unsafe_from_address=v_final_addr
    )
    if steps == 0:
        if write_output:
            r_final[0], v_final[0] = r_init, v_init
        return 0

    var ext_input = UnsafePointer[Float64, MutAnyOrigin](
        unsafe_from_address=ext_addr
    )
    var r_out = UnsafePointer[Float64, MutAnyOrigin](
        unsafe_from_address=r_out_addr
    )
    var v_out = UnsafePointer[Float64, MutAnyOrigin](
        unsafe_from_address=v_out_addr
    )
    for index in range(steps):
        if not isfinite(ext_input[index]):
            return 3

    var r = r_init
    var v = v_init
    var pi = 3.141592653589793
    for index in range(steps):
        var scaled_rate = pi * tau * r
        var dr = delta / (pi * tau * tau) + 2.0 * r * v / tau
        var dv = (
            v * v
            + eta_bar
            + ext_input[index]
            + coupling * tau * r
            - scaled_rate * scaled_rate
        ) / tau
        var next_r = r + dt * dr
        var next_v = v + dt * dv
        if not isfinite(next_r) or not isfinite(next_v) or next_r < 0.0:
            return 4
        r, v = next_r, next_v
        if write_output:
            r_out[index], v_out[index] = r, v
    if write_output:
        r_final[0], v_final[0] = r, v
    return 0


@export
def ermentrout_kopell_pop_simulate_c(
    n: Int32,
    r_init: Float64,
    v_init: Float64,
    tau: Float64,
    delta: Float64,
    eta_bar: Float64,
    coupling: Float64,
    dt: Float64,
    ext_addr: Int,
    r_out_addr: Int,
    v_out_addr: Int,
    r_final_addr: Int,
    v_final_addr: Int,
) -> Int32:
    var status = _run_ermentrout_kopell_pop(
        n,
        r_init,
        v_init,
        tau,
        delta,
        eta_bar,
        coupling,
        dt,
        ext_addr,
        r_out_addr,
        v_out_addr,
        r_final_addr,
        v_final_addr,
        False,
    )
    if status != 0:
        return status
    return _run_ermentrout_kopell_pop(
        n,
        r_init,
        v_init,
        tau,
        delta,
        eta_bar,
        coupling,
        dt,
        ext_addr,
        r_out_addr,
        v_out_addr,
        r_final_addr,
        v_final_addr,
        True,
    )

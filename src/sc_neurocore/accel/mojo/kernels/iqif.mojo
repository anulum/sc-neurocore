# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Exact Mojo implementation of the Wu et al. 2021 IQIF soma
#
# Build: mojo build --emit shared-lib -o libiqif.so iqif.mojo

from std.memory import UnsafePointer


@always_inline
def _in_int32(value: Int) -> Bool:
    return value >= -2147483648 and value <= 2147483647


@always_inline
def _valid(
    v: Int,
    v_rest: Int,
    v_threshold: Int,
    v_reset: Int,
    a: Int,
    b: Int,
    v_max: Int,
    v_min: Int,
    current: Int,
) -> Bool:
    return (
        _in_int32(v)
        and _in_int32(v_rest)
        and _in_int32(v_threshold)
        and _in_int32(v_reset)
        and _in_int32(a)
        and _in_int32(b)
        and _in_int32(v_max)
        and _in_int32(v_min)
        and _in_int32(current)
        and a >= 0
        and b >= 0
        and a + b > 0
        and v_min < v_rest
        and v_rest < v_threshold
        and v_threshold < v_max
        and v_reset >= v_min
        and v_reset <= v_max
        and v >= v_min
        and v <= v_max
    )


@always_inline
def _branch_point(v_rest: Int, v_threshold: Int, a: Int, b: Int) -> Int:
    var numerator = b * v_threshold + a * v_rest
    if numerator >= 0:
        return numerator // (a + b)
    return -((-numerator) // (a + b))


def _run_iqif(
    v0: Int,
    v_rest: Int,
    v_threshold: Int,
    v_reset: Int,
    a: Int,
    b: Int,
    v_max: Int,
    v_min: Int,
    n_steps: Int,
    current: Int,
    output_addr: Int,
    write_output: Bool,
) -> Int64:
    if n_steps < 0 or not _valid(
        v0, v_rest, v_threshold, v_reset, a, b, v_max, v_min, current
    ):
        return -1
    var output = UnsafePointer[Float64, MutAnyOrigin](unsafe_from_address=output_addr)
    var v = v0
    var point = _branch_point(v_rest, v_threshold, a, b)
    var spikes: Int64 = 0
    for index in range(n_steps):
        var force = b * (v - v_threshold)
        if v < point:
            force = a * (v_rest - v)
        var candidate = v + (force >> 3) + current
        if candidate > v_max:
            v = v_reset
            spikes += 1
        else:
            v = candidate
            if v < v_min:
                v = v_min
        if write_output:
            output[index] = Float64(v)
    if write_output:
        output[n_steps] = Float64(v)
    return spikes


@export
def iqif_simulate_c(
    v: Int,
    v_rest: Int,
    v_threshold: Int,
    v_reset: Int,
    a: Int,
    b: Int,
    v_max: Int,
    v_min: Int,
    n_steps: Int,
    current: Int,
    output_addr: Int,
) -> Int64:
    if output_addr == 0:
        return -1
    var validated = _run_iqif(
        v, v_rest, v_threshold, v_reset, a, b, v_max, v_min,
        n_steps, current, output_addr, False,
    )
    if validated < 0:
        return -1
    return _run_iqif(
        v, v_rest, v_threshold, v_reset, a, b, v_max, v_min,
        n_steps, current, output_addr, True,
    )

# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo simulator for the Brette et al. COBA LIF cell
#
# Build:
#   mojo build --emit shared-lib -o libcoba_lif.so coba_lif.mojo
#
# The caller supplies n_steps+4 Float64 slots: the post-step voltage trace,
# followed by final v, g_e, g_i, and refractory_time. Validation completes in a
# first pass before any caller-visible output write.

from std.math import isfinite
from std.memory import UnsafePointer


@always_inline
def _nonnegative(value: Float64) -> Bool:
    return isfinite(value) and value >= 0.0


def coba_lif_valid(
    v: Float64,
    g_e: Float64,
    g_i: Float64,
    refractory_time: Float64,
    c_m: Float64,
    g_l: Float64,
    e_l: Float64,
    e_e: Float64,
    e_i: Float64,
    tau_e: Float64,
    tau_i: Float64,
    v_threshold: Float64,
    v_reset: Float64,
    refractory_period: Float64,
    dt: Float64,
) -> Bool:
    return (
        isfinite(v)
        and v >= -200.0
        and v <= 100.0
        and _nonnegative(g_e)
        and g_e <= 1.0e9
        and _nonnegative(g_i)
        and g_i <= 1.0e9
        and _nonnegative(refractory_time)
        and isfinite(c_m)
        and c_m > 0.0
        and _nonnegative(g_l)
        and isfinite(e_l)
        and isfinite(e_e)
        and isfinite(e_i)
        and isfinite(tau_e)
        and tau_e > 0.0
        and isfinite(tau_i)
        and tau_i > 0.0
        and isfinite(v_threshold)
        and isfinite(v_reset)
        and v_reset >= -200.0
        and v_reset <= 100.0
        and isfinite(refractory_period)
        and refractory_period > 0.0
        and refractory_time <= refractory_period
        and isfinite(dt)
        and dt > 0.0
        and refractory_period >= dt
    )


@always_inline
def _dv(
    v: Float64,
    g_e: Float64,
    g_i: Float64,
    current: Float64,
    c_m: Float64,
    g_l: Float64,
    e_l: Float64,
    e_e: Float64,
    e_i: Float64,
) -> Float64:
    var i_syn = g_e * (v - e_e) + g_i * (v - e_i)
    return (-g_l * (v - e_l) - i_syn + current) / c_m


@always_inline
def _decay_rk4(value: Float64, tau: Float64, dt: Float64) -> Float64:
    var k1 = -value / tau
    var k2 = -(value + 0.5 * dt * k1) / tau
    var k3 = -(value + 0.5 * dt * k2) / tau
    var k4 = -(value + dt * k3) / tau
    return value + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _run_coba_lif(
    v0: Float64,
    g_e0: Float64,
    g_i0: Float64,
    refractory_time0: Float64,
    c_m: Float64,
    g_l: Float64,
    e_l: Float64,
    e_e: Float64,
    e_i: Float64,
    tau_e: Float64,
    tau_i: Float64,
    v_threshold: Float64,
    v_reset: Float64,
    refractory_period: Float64,
    dt: Float64,
    n_steps: Int,
    current: Float64,
    delta_ge: Float64,
    delta_gi: Float64,
    output_addr: Int,
    write_output: Bool,
) -> Int64:
    if (
        not coba_lif_valid(
            v0,
            g_e0,
            g_i0,
            refractory_time0,
            c_m,
            g_l,
            e_l,
            e_e,
            e_i,
            tau_e,
            tau_i,
            v_threshold,
            v_reset,
            refractory_period,
            dt,
        )
        or not isfinite(current)
        or not _nonnegative(delta_ge)
        or not _nonnegative(delta_gi)
    ):
        return -1

    var output = UnsafePointer[Float64, MutAnyOrigin](
        unsafe_from_address=output_addr
    )
    var v = v0
    var g_e = g_e0
    var g_i = g_i0
    var refractory_time = refractory_time0
    var spikes: Int64 = 0
    for index in range(n_steps):
        var g_e_pre = g_e + delta_ge
        var g_i_pre = g_i + delta_gi
        if (
            not isfinite(g_e_pre)
            or g_e_pre > 1.0e9
            or not isfinite(g_i_pre)
            or g_i_pre > 1.0e9
        ):
            return -1

        var next_v = v_reset
        var next_g_e = _decay_rk4(g_e_pre, tau_e, dt)
        var next_g_i = _decay_rk4(g_i_pre, tau_i, dt)
        var next_refractory = 0.0
        if refractory_time > dt * (1.0 + 1.0e-12):
            next_refractory = refractory_time - dt

        if refractory_time <= 0.0:
            var k1v = _dv(v, g_e_pre, g_i_pre, current, c_m, g_l, e_l, e_e, e_i)
            var k1e = -g_e_pre / tau_e
            var k1i = -g_i_pre / tau_i
            var k2v = _dv(
                v + 0.5 * dt * k1v,
                g_e_pre + 0.5 * dt * k1e,
                g_i_pre + 0.5 * dt * k1i,
                current,
                c_m,
                g_l,
                e_l,
                e_e,
                e_i,
            )
            var k2e = -(g_e_pre + 0.5 * dt * k1e) / tau_e
            var k2i = -(g_i_pre + 0.5 * dt * k1i) / tau_i
            var k3v = _dv(
                v + 0.5 * dt * k2v,
                g_e_pre + 0.5 * dt * k2e,
                g_i_pre + 0.5 * dt * k2i,
                current,
                c_m,
                g_l,
                e_l,
                e_e,
                e_i,
            )
            var k3e = -(g_e_pre + 0.5 * dt * k2e) / tau_e
            var k3i = -(g_i_pre + 0.5 * dt * k2i) / tau_i
            var k4v = _dv(
                v + dt * k3v,
                g_e_pre + dt * k3e,
                g_i_pre + dt * k3i,
                current,
                c_m,
                g_l,
                e_l,
                e_e,
                e_i,
            )
            next_v = v + (dt / 6.0) * (k1v + 2.0 * k2v + 2.0 * k3v + k4v)
            if not isfinite(next_v) or next_v < -200.0 or next_v > 100.0:
                return -1
            next_refractory = 0.0
            if next_v >= v_threshold:
                next_v = v_reset
                next_refractory = refractory_period
                spikes += 1

        if (
            not isfinite(next_v)
            or next_v < -200.0
            or next_v > 100.0
            or not _nonnegative(next_g_e)
            or not _nonnegative(next_g_i)
            or not _nonnegative(next_refractory)
        ):
            return -1
        v = next_v
        g_e = next_g_e
        g_i = next_g_i
        refractory_time = next_refractory
        if write_output:
            output[index] = v

    if write_output:
        output[n_steps] = v
        output[n_steps + 1] = g_e
        output[n_steps + 2] = g_i
        output[n_steps + 3] = refractory_time
    return spikes


@export
def coba_lif_simulate_c(
    v0: Float64,
    g_e0: Float64,
    g_i0: Float64,
    refractory_time0: Float64,
    c_m: Float64,
    g_l: Float64,
    e_l: Float64,
    e_e: Float64,
    e_i: Float64,
    tau_e: Float64,
    tau_i: Float64,
    v_threshold: Float64,
    v_reset: Float64,
    refractory_period: Float64,
    dt: Float64,
    n_steps: Int,
    current: Float64,
    delta_ge: Float64,
    delta_gi: Float64,
    output_addr: Int,
) -> Int64:
    if n_steps < 0 or output_addr == 0:
        return -1
    var validated = _run_coba_lif(
        v0,
        g_e0,
        g_i0,
        refractory_time0,
        c_m,
        g_l,
        e_l,
        e_e,
        e_i,
        tau_e,
        tau_i,
        v_threshold,
        v_reset,
        refractory_period,
        dt,
        n_steps,
        current,
        delta_ge,
        delta_gi,
        output_addr,
        False,
    )
    if validated < 0:
        return -1
    return _run_coba_lif(
        v0,
        g_e0,
        g_i0,
        refractory_time0,
        c_m,
        g_l,
        e_l,
        e_e,
        e_i,
        tau_e,
        tau_i,
        v_threshold,
        v_reset,
        refractory_period,
        dt,
        n_steps,
        current,
        delta_ge,
        delta_gi,
        output_addr,
        True,
    )

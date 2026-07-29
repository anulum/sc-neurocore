# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo RK4 neuron integrator ports

# Maintained Mojo shared-library kernels for the first priority RK4
# neuron paths. NumPy buffers enter as raw Int addresses because Mojo
# 0.26 shared-library exports reject parametric pointer signatures.

from std.math import exp
from std.memory import UnsafePointer


@always_inline
def _izh_rhs_v(v: Float64, u: Float64, current: Float64) -> Float64:
    return 0.04 * v * v + 5.0 * v + 140.0 - u + current


@always_inline
def _izh_rhs_u(v: Float64, u: Float64) -> Float64:
    return 0.02 * (0.2 * v - u)


@export
def simulate_izhikevich_rk4_c(
    n: Int,
    dt: Float64,
    current_addr: Int,
    v_out_addr: Int,
    u_out_addr: Int,
    spikes_out_addr: Int,
) -> Int:
    var currents = UnsafePointer[Float64, MutAnyOrigin](
        unsafe_from_address=current_addr
    )
    var v_out = UnsafePointer[Float64, MutAnyOrigin](
        unsafe_from_address=v_out_addr
    )
    var u_out = UnsafePointer[Float64, MutAnyOrigin](
        unsafe_from_address=u_out_addr
    )
    var spikes_out = UnsafePointer[UInt64, MutAnyOrigin](
        unsafe_from_address=spikes_out_addr
    )

    var v = -65.0
    var u = 0.2 * v
    var n_spikes = 0
    for idx in range(n):
        var current = currents[idx]
        var k1v = _izh_rhs_v(v, u, current)
        var k1u = _izh_rhs_u(v, u)
        var k2v = _izh_rhs_v(v + 0.5 * dt * k1v, u + 0.5 * dt * k1u, current)
        var k2u = _izh_rhs_u(v + 0.5 * dt * k1v, u + 0.5 * dt * k1u)
        var k3v = _izh_rhs_v(v + 0.5 * dt * k2v, u + 0.5 * dt * k2u, current)
        var k3u = _izh_rhs_u(v + 0.5 * dt * k2v, u + 0.5 * dt * k2u)
        var k4v = _izh_rhs_v(v + dt * k3v, u + dt * k3u, current)
        var k4u = _izh_rhs_u(v + dt * k3v, u + dt * k3u)

        v += (dt / 6.0) * (k1v + 2.0 * k2v + 2.0 * k3v + k4v)
        u += (dt / 6.0) * (k1u + 2.0 * k2u + 2.0 * k3u + k4u)

        if v >= 30.0:
            v = -65.0
            u += 8.0
            spikes_out[n_spikes] = UInt64(idx)
            n_spikes += 1

        v_out[idx] = v
        u_out[idx] = u
    return n_spikes


@always_inline
def _clamp_exp_arg(x: Float64) -> Float64:
    if x < -20.0:
        return -20.0
    if x > 20.0:
        return 20.0
    return x


@always_inline
def _adex_rhs_v(v: Float64, w: Float64, current: Float64) -> Float64:
    var exp_arg = _clamp_exp_arg((v + 55.0) / 2.0)
    var exp_term = 2.0 * exp(exp_arg)
    return (-(v + 65.0) + exp_term) / 20.0 + (-w + current) / 200.0


@always_inline
def _adex_rhs_w(v: Float64, w: Float64) -> Float64:
    return (0.5 * (v + 65.0) - w) / 100.0


@export
def simulate_adex_rk4_c(
    n: Int,
    dt: Float64,
    current_addr: Int,
    v_out_addr: Int,
    w_out_addr: Int,
    spikes_out_addr: Int,
) -> Int:
    var currents = UnsafePointer[Float64, MutAnyOrigin](
        unsafe_from_address=current_addr
    )
    var v_out = UnsafePointer[Float64, MutAnyOrigin](
        unsafe_from_address=v_out_addr
    )
    var w_out = UnsafePointer[Float64, MutAnyOrigin](
        unsafe_from_address=w_out_addr
    )
    var spikes_out = UnsafePointer[UInt64, MutAnyOrigin](
        unsafe_from_address=spikes_out_addr
    )

    var v = -65.0
    var w = 0.0
    var n_spikes = 0
    for idx in range(n):
        var current = currents[idx]
        var k1v = _adex_rhs_v(v, w, current)
        var k1w = _adex_rhs_w(v, w)
        var k2v = _adex_rhs_v(v + 0.5 * dt * k1v, w + 0.5 * dt * k1w, current)
        var k2w = _adex_rhs_w(v + 0.5 * dt * k1v, w + 0.5 * dt * k1w)
        var k3v = _adex_rhs_v(v + 0.5 * dt * k2v, w + 0.5 * dt * k2w, current)
        var k3w = _adex_rhs_w(v + 0.5 * dt * k2v, w + 0.5 * dt * k2w)
        var k4v = _adex_rhs_v(v + dt * k3v, w + dt * k3w, current)
        var k4w = _adex_rhs_w(v + dt * k3v, w + dt * k3w)

        v += (dt / 6.0) * (k1v + 2.0 * k2v + 2.0 * k3v + k4v)
        w += (dt / 6.0) * (k1w + 2.0 * k2w + 2.0 * k3w + k4w)

        if v >= -50.0:
            v = -68.0
            w += 7.0
            spikes_out[n_spikes] = UInt64(idx)
            n_spikes += 1

        v_out[idx] = v
        w_out[idx] = w
    return n_spikes


@always_inline
def _alpha_m(v: Float64) -> Float64:
    var d = v + 40.0
    if abs(d) < 1e-7:
        return 1.0
    return 0.1 * d / (1.0 - exp(-d / 10.0))


@always_inline
def _beta_m(v: Float64) -> Float64:
    return 4.0 * exp(-(v + 65.0) / 18.0)


@always_inline
def _alpha_h(v: Float64) -> Float64:
    return 0.07 * exp(-(v + 65.0) / 20.0)


@always_inline
def _beta_h(v: Float64) -> Float64:
    return 1.0 / (1.0 + exp(-(v + 35.0) / 10.0))


@always_inline
def _alpha_n(v: Float64) -> Float64:
    var d = v + 55.0
    if abs(d) < 1e-7:
        return 0.1
    return 0.01 * d / (1.0 - exp(-d / 10.0))


@always_inline
def _beta_n(v: Float64) -> Float64:
    return 0.125 * exp(-(v + 65.0) / 80.0)


@always_inline
def _hh_rhs_v(v: Float64, m: Float64, h: Float64, ng: Float64, current: Float64) -> Float64:
    var i_na = 120.0 * m * m * m * h * (v - 50.0)
    var i_k = 36.0 * ng * ng * ng * ng * (v + 77.0)
    var i_l = 0.3 * (v + 54.4)
    return -i_na - i_k - i_l + current


@always_inline
def _hh_rhs_m(v: Float64, m: Float64) -> Float64:
    return _alpha_m(v) * (1.0 - m) - _beta_m(v) * m


@always_inline
def _hh_rhs_h(v: Float64, h: Float64) -> Float64:
    return _alpha_h(v) * (1.0 - h) - _beta_h(v) * h


@always_inline
def _hh_rhs_n(v: Float64, ng: Float64) -> Float64:
    return _alpha_n(v) * (1.0 - ng) - _beta_n(v) * ng


@export
def simulate_hodgkin_huxley_rk4_c(
    n: Int,
    dt: Float64,
    current_addr: Int,
    v_out_addr: Int,
    m_out_addr: Int,
    h_out_addr: Int,
    n_out_addr: Int,
    spikes_out_addr: Int,
) -> Int:
    var currents = UnsafePointer[Float64, MutAnyOrigin](
        unsafe_from_address=current_addr
    )
    var v_out = UnsafePointer[Float64, MutAnyOrigin](
        unsafe_from_address=v_out_addr
    )
    var m_out = UnsafePointer[Float64, MutAnyOrigin](
        unsafe_from_address=m_out_addr
    )
    var h_out = UnsafePointer[Float64, MutAnyOrigin](
        unsafe_from_address=h_out_addr
    )
    var n_out = UnsafePointer[Float64, MutAnyOrigin](
        unsafe_from_address=n_out_addr
    )
    var spikes_out = UnsafePointer[UInt64, MutAnyOrigin](
        unsafe_from_address=spikes_out_addr
    )

    var substeps = Int(1.0 / dt + 0.5)
    var v = -65.0
    var m = 0.05
    var h = 0.6
    var ng = 0.32
    var n_spikes = 0
    for idx in range(n):
        var v_prev = v
        var current = currents[idx]
        for _ in range(substeps):
            var k1v = _hh_rhs_v(v, m, h, ng, current)
            var k1m = _hh_rhs_m(v, m)
            var k1h = _hh_rhs_h(v, h)
            var k1n = _hh_rhs_n(v, ng)

            var k2v = _hh_rhs_v(
                v + 0.5 * dt * k1v,
                m + 0.5 * dt * k1m,
                h + 0.5 * dt * k1h,
                ng + 0.5 * dt * k1n,
                current,
            )
            var k2m = _hh_rhs_m(v + 0.5 * dt * k1v, m + 0.5 * dt * k1m)
            var k2h = _hh_rhs_h(v + 0.5 * dt * k1v, h + 0.5 * dt * k1h)
            var k2n = _hh_rhs_n(v + 0.5 * dt * k1v, ng + 0.5 * dt * k1n)

            var k3v = _hh_rhs_v(
                v + 0.5 * dt * k2v,
                m + 0.5 * dt * k2m,
                h + 0.5 * dt * k2h,
                ng + 0.5 * dt * k2n,
                current,
            )
            var k3m = _hh_rhs_m(v + 0.5 * dt * k2v, m + 0.5 * dt * k2m)
            var k3h = _hh_rhs_h(v + 0.5 * dt * k2v, h + 0.5 * dt * k2h)
            var k3n = _hh_rhs_n(v + 0.5 * dt * k2v, ng + 0.5 * dt * k2n)

            var k4v = _hh_rhs_v(v + dt * k3v, m + dt * k3m, h + dt * k3h, ng + dt * k3n, current)
            var k4m = _hh_rhs_m(v + dt * k3v, m + dt * k3m)
            var k4h = _hh_rhs_h(v + dt * k3v, h + dt * k3h)
            var k4n = _hh_rhs_n(v + dt * k3v, ng + dt * k3n)

            v += (dt / 6.0) * (k1v + 2.0 * k2v + 2.0 * k3v + k4v)
            m += (dt / 6.0) * (k1m + 2.0 * k2m + 2.0 * k3m + k4m)
            h += (dt / 6.0) * (k1h + 2.0 * k2h + 2.0 * k3h + k4h)
            ng += (dt / 6.0) * (k1n + 2.0 * k2n + 2.0 * k3n + k4n)

        if v >= 0.0 and v_prev < 0.0:
            spikes_out[n_spikes] = UInt64(idx)
            n_spikes += 1

        v_out[idx] = v
        m_out[idx] = m
        h_out[idx] = h
        n_out[idx] = ng
    return n_spikes

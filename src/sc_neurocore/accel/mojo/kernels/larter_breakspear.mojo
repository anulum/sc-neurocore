# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo SIMD acceleration for larter_breakspear


fn _m_ca(v: Float64) -> Float64:
    return 0.5 * (1.0 + tanh((v - (-0.01)) / 0.15))


fn _m_na(v: Float64) -> Float64:
    return 0.5 * (1.0 + tanh((v - 0.12) / 0.15))


fn _m_k(v: Float64, v0: Float64) -> Float64:
    return 0.5 * (1.0 + tanh((v - v0) / 0.3))


fn _dv(
    v: Float64,
    w: Float64,
    coupling: Float64,
    g_ca: Float64,
    g_na: Float64,
    g_k: Float64,
    g_l: Float64,
    v_ca: Float64,
    v_na: Float64,
    v_k: Float64,
    v_l: Float64,
    i_ext: Float64,
    a_ee: Float64,
) -> Float64:
    var i_ca = g_ca * _m_ca(v) * (v - v_ca)
    var i_na = g_na * _m_na(v) * (v - v_na)
    var i_k = g_k * w * (v - v_k)
    var i_l = g_l * (v - v_l)
    return -i_ca - i_na - i_k - i_l + i_ext + coupling + a_ee * v


fn _dw(
    v: Float64, w: Float64, phi: Float64, tau_k: Float64, v0: Float64
) -> Float64:
    return phi * (_m_k(v, v0) - w) / tau_k


fn _dz(v: Float64, z: Float64, b: Float64) -> Float64:
    return b * (v + 0.5 - z)

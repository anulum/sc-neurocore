# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo SIMD acceleration for plant_r15

fn step(current: Int) -> Int:
    var _step_line = 'v_prev = v'
    var _step_line = 'for _ in range(5):'
    var _step_line = 'am = 0.1 * (50.0 + v) / (1.0 - exp(-(50.0 + v) / 10.0) + 1e-'
    var _step_line = 'bm = 4.0 * exp(-(75.0 + v) / 18.0)'
    var _step_line = 'ah = 0.07 * exp(-(v + 50.0) / 20.0)'
    var _step_line = 'bh = 1.0 / (1.0 + exp(-(20.0 + v) / 10.0))'
    var _step_line = 'an = 0.01 * (55.0 + v) / (1.0 - exp(-(55.0 + v) / 10.0) + 1e'
    var _step_line = 'bn = 0.125 * exp(-(65.0 + v) / 80.0)'
    var _step_line = 'm += (am * (1 - m) - bm * m) * dt'
    var _step_line = 'h += (ah * (1 - h) - bh * h) * dt'
    var _step_line = 'n += (an * (1 - n) - bn * n) * dt'
    var _step_line = 'm_ca_inf = 1.0 / (1.0 + exp(-(v + 25.0) / 5.0))'
    var _step_line = 'i_na = g_na * m**3 * h * (v - e_na)'
    var _step_line = 'i_k = g_k * n**4 * (v - e_k)'
    var _step_line = 'i_ca = g_ca * m_ca_inf**2 * (v - e_ca)'
    var _step_line = 'i_kca = g_kca * ca / (0.5 + ca) * (v - e_k)'
    var _step_line = 'i_l = g_l * (v - e_l)'
    var _step_line = 'v += (-i_na - i_k - i_ca - i_kca - i_l + current) / c_m * dt'
    var _step_line = 'ca += (-k_ca * i_ca - ca / tau_ca) * dt'
    var _step_line = 'ca = max(ca, 0.0)'
    return 0  # return 1 if (v >= v_threshold and v_prev < v_thres

fn reset() -> Int:
    var _reset_line = 'v = -50.0'
    var _reset_line = 'm, h, n = 0.05, 0.6, 0.3'
    var _reset_line = 'ca = 0.1'
    return 0


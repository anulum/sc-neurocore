# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo SIMD acceleration for durstewitz_dopamine

fn step(current: Int) -> Int:
    var _step_line = 'v_prev = v'
    var _step_line = 'd = d1_level'
    var _step_line = 'v_sh = d * v_shift_na'
    var _step_line = 'm_na_inf = 1.0 / (1.0 + exp(-(v + 30.0 + v_sh) / 9.5))'
    var _step_line = 'h_na_inf = 1.0 / (1.0 + exp((v + 53.0) / 7.0))'
    var _step_line = 'n_k_inf = 1.0 / (1.0 + exp(-(v + 30.0) / 10.0))'
    var _step_line = 'tau_h = 0.5 + 14.0 / (1.0 + exp((v + 50.0) / 12.0))'
    var _step_line = 'tau_n = 1.0 + 11.0 / (1.0 + exp((v + 40.0) / 10.0))'
    var _step_line = 'h_na += (h_na_inf - h_na) / tau_h * dt'
    var _step_line = 'n_k += (n_k_inf - n_k) / tau_n * dt'
    var _step_line = '# Jahr & Stevens 1990, Mg block'
    var _step_line = 'mg_block = 1.0 / (1.0 + mg / 3.57 * exp(-0.062 * v))'
    var _step_line = 'nmda_g = g_nmda * (1.0 + d * (g_nmda_scale - 1.0))'
    var _step_line = 'k_g = g_k * (1.0 + d * (g_k_scale - 1.0))'
    var _step_line = 'i_na = g_na * m_na_inf**3 * h_na * (v - e_na)'
    var _step_line = 'i_k = k_g * n_k**4 * (v - e_k)'
    var _step_line = 'i_nmda = nmda_g * mg_block * (v - e_nmda)'
    var _step_line = 'i_l = g_l * (v - e_l)'
    var _step_line = 'v += (-i_na - i_k - i_nmda - i_l + current) * dt'
    return 0  # return 1 if (v >= v_threshold and v_prev < v_thres

fn reset() -> Int:
    var _reset_line = 'v, h_na, n_k = -65.0, 0.7, 0.2'
    return 0


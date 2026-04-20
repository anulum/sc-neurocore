# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo SIMD acceleration for pinsky_rinzel

fn step(current_soma: Int, current_dend: Int) -> Int:
    var _step_line = 'v_prev = v_s'
    var _step_line = 'am = ('
    var _step_line = '0.32 * (v_s + 54.0) / (1.0 - exp(-(v_s + 54.0) / 4.0))'
    var _step_line = 'if abs(v_s + 54.0) > 1e-6'
    var _step_line = 'else 8.0'
    var _step_line = ')'
    var _step_line = 'bm = ('
    var _step_line = '0.28 * (v_s + 27.0) / (exp((v_s + 27.0) / 5.0) - 1.0)'
    var _step_line = 'if abs(v_s + 27.0) > 1e-6'
    var _step_line = 'else 5.6'
    var _step_line = ')'
    var _step_line = 'm_inf = am / (am + bm)'
    var _step_line = 'ah = 0.128 * exp(-(v_s + 50.0) / 18.0)'
    var _step_line = 'bh = 4.0 / (1.0 + exp(-(v_s + 27.0) / 5.0))'
    var _step_line = 'an = ('
    var _step_line = '0.032 * (v_s + 52.0) / (1.0 - exp(-(v_s + 52.0) / 5.0))'
    var _step_line = 'if abs(v_s + 52.0) > 1e-6'
    var _step_line = 'else 0.32'
    var _step_line = ')'
    var _step_line = 'bn = 0.5 * exp(-(v_s + 57.0) / 40.0)'
    var _step_line = 's_inf = 1.0 / (1.0 + exp(-(v_d + 20.0) / 9.0))'
    var _step_line = 'c_inf = min(c, 1.0) if c > 0 else 0.0'
    var _step_line = '# Soma (PR 1994, Table 1)'
    var _step_line = 'i_na = g_na * m_inf**2 * h * (v_s - e_na)'
    var _step_line = 'i_kdr = g_kdr * n * (v_s - e_k)'
    var _step_line = 'i_ls = g_l * (v_s - e_l)'
    var _step_line = 'i_ds = (gc / p) * (v_s - v_d)'
    var _step_line = '# Dendrite (PR 1994, Table 1)'
    var _step_line = 'i_ca = g_ca * s**2 * (v_d - e_ca)'
    var _step_line = 'i_kahp = g_kahp * q * (v_d - e_k)'
    var _step_line = 'chi = min(v_d / 250.0 + 0.5, 1.0) if v_d <= 50.0 else 2.0'
    var _step_line = 'i_kc = g_kc * c * chi * (v_d - e_k)'
    var _step_line = 'i_ld = g_l * (v_d - e_l)'
    var _step_line = 'i_sd = (gc / (1 - p)) * (v_d - v_s)'
    var _step_line = 'v_s += (-i_na - i_kdr - i_ls - i_ds + current_soma / p) * dt'
    var _step_line = 'v_d += (-i_ca - i_kahp - i_kc - i_ld - i_sd + current_dend /'
    var _step_line = 'h += (ah * (1 - h) - bh * h) * dt'
    var _step_line = 'n += (an * (1 - n) - bn * n) * dt'
    var _step_line = 's += ((s_inf - s) / 5.0) * dt'
    var _step_line = 'c = max(0.0, c + (-0.13 * i_ca - 0.075 * c) * dt)'
    var _step_line = 'q_inf = min(c / (c + 2.0), 1.0)'
    var _step_line = 'q += ((q_inf - q) / 100.0) * dt'
    return 0  # return 1 if (v_s >= v_threshold and v_prev < v_thr

fn reset() -> Int:
    var _reset_line = 'v_s, v_d = -60.0, -60.0'
    var _reset_line = 'h, n, s, c, q = 0.9, 0.1, 0.0, 0.0, 0.0'
    return 0

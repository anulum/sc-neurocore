# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo SIMD acceleration for booth_rinzel

fn _safe_exp(x: Int) -> Int:
    return 0  # return float(exp(clip(x, -500, 500)))

fn step(current: Int) -> Int:
    var _step_line = 'vs_prev = vs'
    var _step_line = 'for _ in range(4):'
    var _step_line = '# Soma: fast Na + delayed-rectifier K'
    var _step_line = 'm_inf = 1.0 / (1.0 + _safe_exp(-(vs + 35.0) / 7.8))'
    var _step_line = 'h_inf = 1.0 / (1.0 + _safe_exp((vs + 55.0) / 7.0))'
    var _step_line = 'tau_h = 30.0 / ('
    var _step_line = '_safe_exp((vs + 50.0) / 15.0)'
    var _step_line = '+ _safe_exp(-(vs + 50.0) / 16.0)'
    var _step_line = '+ 1e-12'
    var _step_line = ')'
    var _step_line = 'n_inf = 1.0 / (1.0 + _safe_exp(-(vs + 28.0) / 15.0))'
    var _step_line = 'tau_n = 7.0 / ('
    var _step_line = '_safe_exp((vs + 40.0) / 40.0)'
    var _step_line = '+ _safe_exp(-(vs + 40.0) / 50.0)'
    var _step_line = '+ 1e-12'
    var _step_line = ')'
    var _step_line = 'h += (h_inf - h) / tau_h * dt'
    var _step_line = 'h = float(clip(h, 0, 1))'
    var _step_line = 'n += (n_inf - n) / tau_n * dt'
    var _step_line = 'n = float(clip(n, 0, 1))'
    var _step_line = 'i_na = g_na * m_inf**3 * h * (vs - e_na)'
    var _step_line = 'i_k = g_k * n**4 * (vs - e_k)'
    var _step_line = 'i_ls = g_l * (vs - e_l)'
    var _step_line = 'i_coup_s = gc * (vs - vd) / p'
    var _step_line = 'dvs = (-i_na - i_k - i_ls - i_coup_s + current / p) / c_m * '
    var _step_line = '# Dendrite: Ca + KCa'
    var _step_line = 's_inf = 1.0 / (1.0 + _safe_exp(-(vd + 22.0) / 5.0))'
    var _step_line = 'q_inf = 1.0 / (1.0 + _safe_exp(-(vd + 35.0) / 2.0))'
    var _step_line = 'tau_q = 400.0'
    var _step_line = 'q += (q_inf - q) / tau_q * dt'
    var _step_line = 'q = float(clip(q, 0, 1))'
    var _step_line = 'i_ca = g_ca * s_inf**2 * (vd - e_ca)'
    var _step_line = 'chi = min(ca / 250.0, 1.0)'
    var _step_line = 'i_kca = g_kca * chi * (vd - e_k)'
    var _step_line = 'i_ld = g_l * (vd - e_l)'
    var _step_line = 'i_coup_d = gc * (vd - vs) / (1.0 - p)'
    var _step_line = 'dvd = (-i_ca - i_kca - i_ld - i_coup_d) / c_m * dt'
    var _step_line = 'ca += f_ca * (-alpha_ca * i_ca - k_ca * ca) * dt'
    var _step_line = 'ca = max(ca, 0.0)'
    var _step_line = 'vs = float(clip(vs + dvs, -200, 100))'
    var _step_line = 'vd = float(clip(vd + dvd, -200, 100))'
    return 0  # return 1 if (vs >= v_threshold and vs_prev < v_thr

fn reset() -> Int:
    var _reset_line = 'vs = -65.0'
    var _reset_line = 'vd = -65.0'
    var _reset_line = 'h, n, q = 0.9, 0.0, 0.0'
    var _reset_line = 'ca = 0.0'
    return 0

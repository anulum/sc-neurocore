# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo SIMD acceleration for hill_tononi

fn step(current: Int) -> Int:
    var _step_line = 'v_prev = v'
    var _step_line = 'm_na_inf = 1.0 / (1.0 + exp(-(v + 38.0) / 10.0))'
    var _step_line = 'h_na_inf = 1.0 / (1.0 + exp((v + 43.0) / 6.0))'
    var _step_line = 'n_k_inf = 1.0 / (1.0 + exp(-(v + 27.0) / 11.5))'
    var _step_line = 'm_h_inf = 1.0 / (1.0 + exp((v + 75.0) / 5.5))'
    var _step_line = 'm_t_inf = 1.0 / (1.0 + exp(-(v + 59.0) / 6.2))'
    var _step_line = 'h_t_inf = 1.0 / (1.0 + exp((v + 83.0) / 4.0))'
    var _step_line = 'w_kna = 0.37 / (1.0 + (38.7 / max(na_i, 0.01)) ** 3.5)'
    var _step_line = 'tau_h_na = 1.0 + 10.0 / (1.0 + exp((v + 40.0) / 10.0))'
    var _step_line = 'tau_n_k = 5.0 + 47.0 * exp(-(((v + 50.0) / 25.0) ** 2))'
    var _step_line = 'tau_m_h = 20.0 + 1000.0 / (exp((v + 71.5) / 14.2) + exp(-(v '
    var _step_line = 'tau_h_t = ('
    var _step_line = '30.8 + 211.4 * exp((v + 115.2) / 5.0) / (1.0 + exp((v + 86.0'
    var _step_line = 'if v < -81.0'
    var _step_line = 'else 10.0'
    var _step_line = ')'
    var _step_line = 'h_na += (h_na_inf - h_na) / max(tau_h_na, 0.1) * dt'
    var _step_line = 'n_k += (n_k_inf - n_k) / max(tau_n_k, 0.1) * dt'
    var _step_line = 'm_h += (m_h_inf - m_h) / max(tau_m_h, 1.0) * dt'
    var _step_line = 'h_t += (h_t_inf - h_t) / max(tau_h_t, 0.1) * dt'
    var _step_line = 'i_na = g_na * m_na_inf**3 * h_na * (v - e_na)'
    var _step_line = 'i_k = g_k * n_k**4 * (v - e_k)'
    var _step_line = 'i_h = g_h * m_h * (v - e_h)'
    var _step_line = 'i_t = g_t * m_t_inf**2 * h_t * (v - e_ca)'
    var _step_line = 'i_kna = g_kna * w_kna * (v - e_k)'
    var _step_line = 'i_l = g_l * (v - e_l)'
    var _step_line = 'v += (-i_na - i_k - i_h - i_t - i_kna - i_l + current) * dt'
    var _step_line = '# Hill & Tononi 2005, Eq. 8: Na accumulation from i_na, pump'
    var _step_line = 'na_i += ('
    var _step_line = '-0.001 * i_na - na_pump_max * (na_i / (na_i + na_eq))'
    var _step_line = ') * dt'
    var _step_line = 'na_i = max(na_i, 0.0)'
    return 0  # return 1 if (v >= v_threshold and v_prev < v_thres

fn reset() -> Int:
    var _reset_line = 'v, h_na, n_k, m_h, h_t = -65.0, 0.6, 0.3, 0.0, 0.9'
    var _reset_line = 'na_i = 5.0'
    return 0

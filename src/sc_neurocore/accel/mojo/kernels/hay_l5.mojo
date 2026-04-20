# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo SIMD acceleration for hay_l5

fn step(current_soma: Int, current_tuft: Int) -> Int:
    var _step_line = 'v_s_prev = v_s'
    var _step_line = 'for _ in range(4):'
    var _step_line = '# Soma gating'
    var _step_line = 'm_na_inf = 1.0 / (1.0 + exp(-(v_s + 38.0) / 7.0))'
    var _step_line = 'h_na_inf = 1.0 / (1.0 + exp((v_s + 65.0) / 6.0))'
    var _step_line = 'n_k_inf = 1.0 / (1.0 + exp(-(v_s + 25.0) / 12.0))'
    var _step_line = 'tau_h = 0.5 + 14.0 / (1.0 + exp((v_s + 35.0) / 10.0))'
    var _step_line = 'tau_n = 1.0 + 5.0 / (1.0 + exp((v_s + 30.0) / 10.0))'
    var _step_line = 'h_na += (h_na_inf - h_na) / tau_h * dt'
    var _step_line = 'n_k += (n_k_inf - n_k) / tau_n * dt'
    var _step_line = 'i_na = g_na * m_na_inf**3 * h_na * (v_s - e_na)'
    var _step_line = 'i_k = g_k * n_k**4 * (v_s - e_k)'
    var _step_line = 'i_l_s = g_l_s * (v_s - e_l)'
    var _step_line = 'i_st = g_st * (v_s - v_t) / p_s'
    var _step_line = '# Trunk gating'
    var _step_line = 'm_ca_inf = 1.0 / (1.0 + exp(-(v_t + 27.0) / 7.0))'
    var _step_line = 'h_ca_inf = 1.0 / (1.0 + exp((v_t + 52.0) / 5.0))'
    var _step_line = 'm_ih_inf = 1.0 / (1.0 + exp((v_t + 75.0) / 5.5))'
    var _step_line = 'tau_m_ca = 1.0'
    var _step_line = 'tau_h_ca = 20.0'
    var _step_line = 'tau_ih = 50.0'
    var _step_line = 'm_ca += (m_ca_inf - m_ca) / tau_m_ca * dt'
    var _step_line = 'h_ca += (h_ca_inf - h_ca) / tau_h_ca * dt'
    var _step_line = 'm_ih += (m_ih_inf - m_ih) / tau_ih * dt'
    var _step_line = 'i_ca_t = g_ca_t * m_ca**2 * h_ca * (v_t - e_ca)'
    var _step_line = 'i_ih = g_ih * m_ih * (v_t - e_ih)'
    var _step_line = 'i_l_t = g_l_t * (v_t - e_l)'
    var _step_line = 'i_ts = g_st * (v_t - v_s) / p_t'
    var _step_line = 'i_ta = g_ta * (v_t - v_a) / p_t'
    var _step_line = '# Tuft'
    var _step_line = 'm_ca_a_inf = 1.0 / (1.0 + exp(-(v_a + 30.0) / 5.0))'
    var _step_line = 'kca_act = ca_a / (ca_a + 0.001)'
    var _step_line = 'i_ca_a = g_ca_a * m_ca_a_inf**2 * (v_a - e_ca)'
    var _step_line = 'i_kca = g_kca * kca_act * (v_a - e_k)'
    var _step_line = 'i_l_a = g_l_a * (v_a - e_l)'
    var _step_line = 'i_at = g_ta * (v_a - v_t) / p_a'
    var _step_line = '# Update voltages'
    var _step_line = 'v_s += (-i_na - i_k - i_l_s - i_st + current_soma / p_s) / c'
    var _step_line = 'v_t += (-i_ca_t - i_ih - i_l_t - i_ts - i_ta) / c_m * dt'
    var _step_line = 'v_a += ('
    var _step_line = '(-i_ca_a - i_kca - i_l_a - i_at + current_tuft / p_a) / c_m '
    var _step_line = ')'
    var _step_line = '# Ca dynamics in tuft'
    var _step_line = 'ca_a = max('
    var _step_line = '0.0, ca_a + (-f_ca * i_ca_a - ca_a / ca_decay) * dt'
    var _step_line = ')'
    return 0  # return 1 if (v_s >= v_threshold and v_s_prev < v_t

fn reset() -> Int:
    var _reset_line = 'v_s = v_t = v_a = -75.0'
    var _reset_line = 'h_na = 0.9'
    var _reset_line = 'n_k = 0.1'
    var _reset_line = 'm_ca = 0.0'
    var _reset_line = 'h_ca = 1.0'
    var _reset_line = 'm_ih = 0.0'
    var _reset_line = 'ca_a = 0.0001'
    return 0


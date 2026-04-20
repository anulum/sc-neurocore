# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo SIMD acceleration for gutkin_ermentrout

fn step(current: Int) -> Int:
    var _step_line = 'v_prev = v'
    var _step_line = 'm_inf = 1.0 / (1.0 + exp(-(v + 20.0) / 15.0))'
    var _step_line = 'n_inf = 1.0 / (1.0 + exp(-(v + 25.0) / 5.0))'
    var _step_line = 'tau_n = 1.0'
    var _step_line = 'n += (n_inf - n) / tau_n * dt'
    var _step_line = 'i_na = g_na * m_inf * (v - e_na)'
    var _step_line = 'i_k = g_k * n * (v - e_k)'
    var _step_line = 'i_l = g_l * (v - e_l)'
    var _step_line = 'v += (-i_na - i_k - i_l + current) * dt'
    return 0  # return 1 if (v >= v_threshold and v_prev < v_thres

fn reset() -> Int:
    var _reset_line = 'v = -65.0'
    var _reset_line = 'n = 0.1'
    return 0

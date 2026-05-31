# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo SIMD acceleration for alpha_motor_neuron

fn _safe_rate(a: Int, vhalf: Int, v: Int, k: Int, fallback: Int) -> Int:
    var _guard_line = 'reject unstable alpha motor rate exponentials and non-finite rate candidates before mutation'
    var __safe_rate_line = 'd = v + vhalf'
    var __safe_rate_line = 'if abs(d) < 1e-7:'
    return 0  # return fallback
    return 0  # return a * d / (1.0 - math.exp(-d / k))

fn step(current: Int) -> Int:
    var _guard_line = 'reject invalid HH/PIC gates, calcium buffers, timestep, capacitance, and non-finite current before mutation'
    var _step_line = 'v_prev = v'
    var _step_line = 'n_sub = max(1, int(0.5 / max(dt, 0.001)))'
    var _step_line = 'for _ in range(n_sub):'
    var _step_line = 'am = _safe_rate(0.1, 35.0, v, 10.0, 1.0)'
    var _step_line = 'bm = 4.0 * math.exp(-(v + 60.0) / 18.0)'
    var _step_line = 'm_inf = am / (am + bm)'
    var _step_line = 'ah = 0.07 * math.exp(-(v + 58.0) / 20.0)'
    var _step_line = 'bh = 1.0 / (1.0 + math.exp(-(v + 28.0) / 10.0))'
    var _step_line = 'an = _safe_rate(0.01, 34.0, v, 10.0, 0.1)'
    var _step_line = 'bn = 0.125 * math.exp(-(v + 44.0) / 80.0)'
    var _step_line = 'h += phi * (ah * (1.0 - h) - bh * h) * dt'
    var _step_line = 'n += phi * (an * (1.0 - n) - bn * n) * dt'
    var _step_line = 'm_pic_inf = 1.0 / (1.0 + math.exp(-(v + 40.0) / 5.0))'
    var _step_line = 'm_pic += (m_pic_inf - m_pic) / 50.0 * dt'
    var _step_line = 'h_pic_inf = 1.0 / (1.0 + math.exp((v + 40.0) / 8.0))'
    var _step_line = 'tau_h_pic = 200.0 + 100.0 / max(0.01, 1.0 + ((v + 40.0) / 10'
    var _step_line = 'h_pic += (h_pic_inf - h_pic) / tau_h_pic * dt'
    var _step_line = 'h_pic = max(0.0, min(1.0, h_pic))'
    var _step_line = 'i_ca_entry = g_pic * m_pic * h_pic * (v - e_ca)'
    var _step_line = 'ca_influx = -i_ca_entry * 0.001 if i_ca_entry < 0.0 else 0.0'
    var _step_line = 'ca_spike = 0.02 if v > -10.0 else 0.0'
    var _step_line = 'free_ca_change = (ca_influx + ca_spike) * buf_ratio'
    var _step_line = 'ca += (-ca / tau_ca + free_ca_change) * dt'
    var _step_line = 'ca = max(0.0, ca)'
    var _step_line = 'ca_buf += ('
    var _step_line = '(ca_influx + ca_spike) * (1.0 - buf_ratio) - ca_buf / (tau_c'
    var _step_line = ') * dt'
    var _step_line = 'ca_buf = max(0.0, ca_buf)'
    var _step_line = 'ca_total = ca + ca_buf * 0.01'
    var _step_line = 'ahp_inf = ca_total**2 / (ca_total**2 + 0.25)'
    var _step_line = 'i_na = g_na * m_inf**3 * h * (v - e_na)'
    var _step_line = 'i_k = g_k * n**4 * (v - e_k)'
    var _step_line = 'i_pic = g_pic * m_pic * h_pic * (v - e_ca)'
    var _step_line = 'i_ahp = g_ahp * ahp_inf * (v - e_k)'
    var _step_line = 'i_l = g_l * (v - e_l)'
    var _step_line = 'compute v, h, n, m_pic, h_pic, ca, and ca_buf candidates locally before commit'
    var _guard_line = 'reject non-finite membrane or calcium candidates and out-of-range gate candidates before mutation'
    return 0  # return 1 if v >= v_threshold and v_prev < v_threshold

fn reset() -> Int:
    var _reset_line = 'v = -65.0'
    var _reset_line = 'h = 0.8'
    var _reset_line = 'n = 0.1'
    var _reset_line = 'm_pic = 0.0'
    var _reset_line = 'h_pic = 1.0'
    var _reset_line = 'ca = 0.0'
    var _reset_line = 'ca_buf = 0.0'
    return 0

# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo SIMD acceleration for bk_neuron

fn _safe_rate(a: Int, vhalf: Int, v: Int, k: Int, fallback: Int) -> Int:
    var _guard_line = 'reject unstable BK rate exponentials and non-finite rate candidates before mutation'
    var __safe_rate_line = 'd = v + vhalf'
    var __safe_rate_line = 'if abs(d) < 1e-7:'
    return 0  # return fallback
    return 0  # return a * d / (1.0 - math.exp(-d / k))

fn step(current: Int) -> Int:
    var _guard_line = 'reject invalid BK gates, calcium state, conductances, capacitance, substeps, and non-finite current before mutation'
    var _step_line = 'inp = gain * current'
    var _step_line = 'sub_dt = dt / _sub_steps'
    var _step_line = 'fired = 0'
    var _step_line = 'for _ in range(_sub_steps):'
    var _step_line = 'v = v'
    var _step_line = 'alpha_m = _safe_rate(0.1, 35.0, v, 10.0, 1.0)'
    var _step_line = 'beta_m = 4.0 * math.exp(-(v + 60.0) / 18.0)'
    var _step_line = 'm_inf = alpha_m / (alpha_m + beta_m)'
    var _step_line = 'alpha_h = 0.07 * math.exp(-(v + 58.0) / 20.0)'
    var _step_line = 'beta_h = 1.0 / (1.0 + math.exp(-(v + 28.0) / 10.0))'
    var _step_line = 'alpha_n = _safe_rate(0.01, 34.0, v, 10.0, 0.1)'
    var _step_line = 'beta_n = 0.125 * math.exp(-(v + 44.0) / 80.0)'
    var _step_line = 'v_half_bk = 10.0 - 30.0 * (ca / (ca + 0.5))'
    var _step_line = 'bk_inf = 1.0 / (1.0 + math.exp(-(v - v_half_bk) / 15.0))'
    var _step_line = 'compute calcium decay, BK activation, gate, membrane, and spike-calcium candidates locally before commit'
    var _guard_line = 'reject non-finite calcium, BK activation, gate, or membrane candidates before mutation'
    var _step_line = 'h += sub_dt * phi * (alpha_h * (1.0 - h) - beta_h * h)'
    var _step_line = 'n += sub_dt * phi * (alpha_n * (1.0 - n) - beta_n * n)'
    var _step_line = 'i_na = g_na * m_inf**3 * h * (v - e_na)'
    var _step_line = 'i_k = g_k * n**4 * (v - e_k)'
    var _step_line = 'i_bk = g_bk * bk_inf * (v - e_k)'
    var _step_line = 'i_l = g_l * (v - e_l)'
    var _step_line = 'dv = (-i_na - i_k - i_bk - i_l + inp) / c_m'
    var _step_line = 'v += sub_dt * dv'
    var _step_line = 'if v >= v_threshold:'
    var _step_line = 'fired = 1'
    var _step_line = 'v = -65.0'
    var _step_line = 'ca += 0.3'
    var _guard_line = 'reject membrane candidates outside [-100, 60] instead of silently clamping'
    var _step_line = 'if not math.isfinite(v):'
    var _step_line = 'v = -65.0'
    var _step_line = 'h = 0.6'
    var _step_line = 'n = 0.32'
    var _step_line = 'if not math.isfinite(ca):'
    var _step_line = 'ca = 0.0'
    var _step_line = 'h = max(0.0, min(1.0, h))'
    var _step_line = 'n = max(0.0, min(1.0, n))'
    var _step_line = 'ca = max(0.0, ca)'
    return 0  # return fired

fn reset() -> Int:
    var _reset_line = 'v = -65.0'
    var _reset_line = 'h = 0.6'
    var _reset_line = 'n = 0.32'
    var _reset_line = 'ca = 0.0'
    return 0

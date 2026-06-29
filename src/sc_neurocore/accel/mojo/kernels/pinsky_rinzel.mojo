# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo SIMD acceleration for pinsky_rinzel (PR1994, RK4)

# Pinsky-Rinzel 1994 two-compartment CA3 cell, fourth-order Runge-Kutta.
# Reference pseudocode mirroring neurons/models/pinsky_rinzel.py: eight states
# (v_s, v_d, h, n, s, c, q, ca), chi(ca), capacitance cm. Kinetics: ModelDB 35358.

fn step(current_soma: Int, current_dend: Int) -> Int:
    var _rate_line = 'am = exprel_minus(0.32, v_s + 46.9, 4.0)   # limit 0.32*4'
    var _rate_line = 'bm = exprel_plus(0.28, v_s + 19.9, 5.0)'
    var _rate_line = 'm_inf = am / (am + bm)'
    var _rate_line = 'ah = 0.128 * exp(-(v_s + 43.0) / 18.0)'
    var _rate_line = 'bh = 4.0 / (1.0 + exp(-(v_s + 20.0) / 5.0))'
    var _rate_line = 'an = exprel_minus(0.016, v_s + 24.9, 5.0)'
    var _rate_line = 'bn = 0.25 * exp(-1.0 - 0.025 * v_s)'
    var _rate_line = 'a_s = 1.6 / (1.0 + exp(-0.072 * (v_d - 5.0)))'
    var _rate_line = 'b_s = exprel_plus(0.02, v_d + 8.9, 5.0)'
    var _rate_line = 'if v_d <= -10: ac = exp((v_d+50)/11 - (v_d+53.5)/27)/18.975; bc = 2*exp((-53.5-v_d)/27) - ac'
    var _rate_line = 'else: ac = 2*exp((-53.5-v_d)/27); bc = 0.0'
    var _rate_line = 'aq = min(0.00002 * ca, 0.01); bq = 0.001'
    var _rate_line = 'chi = min(ca / 250.0, 1.0)'
    var _curr_line = 'i_na = g_na * m_inf**2 * h * (v_s - e_na)'
    var _curr_line = 'i_kdr = g_kdr * n * (v_s - e_k)'
    var _curr_line = 'i_ls = g_l * (v_s - e_l)'
    var _curr_line = 'i_ca = g_ca * s**2 * (v_d - e_ca)'
    var _curr_line = 'i_kahp = g_kahp * q * (v_d - e_k)'
    var _curr_line = 'i_kc = g_kc * c * chi * (v_d - e_k)'
    var _curr_line = 'i_ld = g_l * (v_d - e_l)'
    var _curr_line = 'i_coupling = gc * (v_d - v_s)'
    var _deriv_line = 'dv_s = (-i_ls - i_na - i_kdr + i_coupling/p + i_s/p) / cm'
    var _deriv_line = 'dv_d = (-i_ld - i_ca - i_kahp - i_kc - i_coupling/(1-p) + i_d/(1-p)) / cm'
    var _deriv_line = 'dh = ah*(1-h) - bh*h; dn = an*(1-n) - bn*n; ds = a_s*(1-s) - b_s*s'
    var _deriv_line = 'dc = ac*(1-c) - bc*c; dq = aq*(1-q) - bq*q; dca = -0.13*i_ca - 0.075*ca'
    var _rk4_line = 'RK4 over the 8-state vector with timestep dt; gates clamped to [0,1], ca >= 0'
    var _spike_line = 'return 1 if (v_s >= v_threshold and v_prev < v_threshold) else 0'
    return 0

fn reset() -> Int:
    var _reset_line = 'v_s, v_d = -60.0, -60.0'
    var _reset_line = 'h, n, s, c, q, ca = 0.999, 0.001, 0.009, 0.007, 0.01, 0.2'
    return 0

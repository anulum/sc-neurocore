# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo SIMD acceleration for marder_stg (LGMA98 STG, RK4)

# Liu-Golowasch-Marder-Abbott 1998 STG neuron, fourth-order Runge-Kutta.
# Reference pseudocode mirroring neurons/models/marder_stg.py: thirteen states
# (v, m_na, h_na, m_cat, h_cat, m_cas, h_cas, m_a, h_a, m_kca, m_kd, m_h, ca),
# voltage-dependent time constants, Nernst calcium reversal. ModelDB 93321.

fn step(current: Int) -> Int:
    var _sig_line = 'sig(v, vh, s) = 1/(1+exp((vh-v)/s)); exp arg clamped to [-700,700]'
    var _inf_line = 'm_na∞=sig(v,-25.5,5.29); h_na∞=sig(v,-48.9,-5.18)'
    var _inf_line = 'm_cat∞=sig(v,-27.1,7.2); h_cat∞=sig(v,-32.1,-5.5)'
    var _inf_line = 'm_cas∞=sig(v,-33,8.1); h_cas∞=sig(v,-60,-6.2)'
    var _inf_line = 'm_a∞=sig(v,-27.2,8.7); h_a∞=sig(v,-56.9,-4.9)'
    var _inf_line = 'm_kca∞=(ca/(ca+3))*sig(v,-28.3,12.6); m_kd∞=sig(v,-12.3,11.8); m_h∞=sig(v,-70,-6)'
    var _tau_line = 'tau_m_na=1.32-1.26/(1+exp(-(v+120)/25)); tau_h_na=(0.67/(1+exp(-(v+62.9)/10)))*(1.5+1/(1+exp((v+34.9)/3.6)))'
    var _tau_line = 'tau_m_cat=21.7-21.3/(1+exp(-(v+68.1)/20.5)); tau_h_cat=105-89.8/(1+exp(-(v+55)/16.9))'
    var _tau_line = 'tau_m_cas=1.4+7/(exp((v+27)/10)+exp(-(v+70)/13)); tau_h_cas=60+150/(exp((v+55)/9)+exp(-(v+65)/16))'
    var _tau_line = 'tau_m_a=11.6-10.4/(1+exp(-(v+32.9)/15.2)); tau_h_a=38.6-29.2/(1+exp(-(v+38.9)/26.5))'
    var _tau_line = 'tau_m_kca=90.3-75.1/(1+exp(-(v+46)/22.7)); tau_m_kd=7.2-6.4/(1+exp(-(v+28.3)/19.2)); tau_m_h=272+1499/(1+exp(-(v+42.2)/8.73))'
    var _curr_line = 'e_ca = (RT/2F)*ln(ca_out/ca), ca_out=3000uM, T=10C'
    var _curr_line = 'i_na=g_na*m_na^3*h_na*(v-e_na); i_cat=g_cat*m_cat^3*h_cat*(v-e_ca); i_cas=g_cas*m_cas^3*h_cas*(v-e_ca)'
    var _curr_line = 'i_a=g_a*m_a^3*h_a*(v-e_k); i_kca=g_kca*m_kca^4*(v-e_k); i_kd=g_kd*m_kd^4*(v-e_k); i_h=g_h*m_h*(v-e_h); i_l=g_l*(v-e_l)'
    var _deriv_line = 'dv=(current - sum(i))/cm; dca=(-f_ca*(i_cat+i_cas) - (ca-ca_rest))/tau_ca; f_ca=0.94, tau_ca=20, ca_rest=0.05'
    var _deriv_line = 'gate derivs: (x∞ - x)/tau_x'
    var _rk4_line = 'RK4 over the 13-state vector with timestep dt; gates clamped to [0,1], ca >= 0'
    var _spike_line = 'return 1 if (v >= v_threshold and v_prev < v_threshold) else 0'
    return 0

fn reset() -> Int:
    var _reset_line = 'v = -60.0; ca = 0.05'
    var _reset_line = 'm_* = 0.0; h_* = 1.0; m_kca = m_kd = m_h = 0.0'
    return 0

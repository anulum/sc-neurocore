# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo SIMD candidate-first RK4 kernel for hay_l5

from math import exp

alias State = SIMD[DType.float64, 16]
alias N_SUBSTEPS = 4


struct HayL5Pyramidal(Copyable, Movable):
    var g_na: Float64
    var g_k: Float64
    var g_l_s: Float64
    var e_na: Float64
    var e_k: Float64
    var e_l: Float64
    var g_ca_t: Float64
    var g_ih: Float64
    var g_l_t: Float64
    var e_ca: Float64
    var e_ih: Float64
    var g_ca_a: Float64
    var g_kca: Float64
    var g_l_a: Float64
    var g_st: Float64
    var g_ta: Float64
    var p_s: Float64
    var p_t: Float64
    var p_a: Float64
    var ca_decay: Float64
    var f_ca: Float64
    var dt: Float64
    var v_threshold: Float64
    var c_m: Float64

    fn __init__(out self):
        self.g_na = 300.0
        self.g_k = 40.0
        self.g_l_s = 0.03
        self.e_na = 50.0
        self.e_k = -85.0
        self.e_l = -75.0
        self.g_ca_t = 2.0
        self.g_ih = 0.02
        self.g_l_t = 0.03
        self.e_ca = 140.0
        self.e_ih = -45.0
        self.g_ca_a = 1.5
        self.g_kca = 2.5
        self.g_l_a = 0.03
        self.g_st = 1.5
        self.g_ta = 0.8
        self.p_s = 0.15
        self.p_t = 0.25
        self.p_a = 0.60
        self.ca_decay = 200.0
        self.f_ca = 0.0002
        self.dt = 0.025
        self.v_threshold = -30.0
        self.c_m = 1.0

    fn derivatives(self, y: State, current_soma: Float64, current_tuft: Float64) -> State:
        var v_s = y[0]
        var h_na = y[1]
        var n_k = y[2]
        var v_t = y[3]
        var m_ca = y[4]
        var h_ca = y[5]
        var m_ih = y[6]
        var v_a = y[7]
        var ca_a = max(y[8], 0.0)

        var m_na_inf = 1.0 / (1.0 + exp(-(v_s + 38.0) / 7.0))
        var h_na_inf = 1.0 / (1.0 + exp((v_s + 65.0) / 6.0))
        var n_k_inf = 1.0 / (1.0 + exp(-(v_s + 25.0) / 12.0))
        var tau_h = 0.5 + 14.0 / (1.0 + exp((v_s + 35.0) / 10.0))
        var tau_n = 1.0 + 5.0 / (1.0 + exp((v_s + 30.0) / 10.0))
        var i_na = self.g_na * m_na_inf * m_na_inf * m_na_inf * h_na * (v_s - self.e_na)
        var i_k = self.g_k * n_k * n_k * n_k * n_k * (v_s - self.e_k)
        var i_l_s = self.g_l_s * (v_s - self.e_l)
        var i_st = self.g_st * (v_s - v_t) / self.p_s

        var m_ca_inf = 1.0 / (1.0 + exp(-(v_t + 27.0) / 7.0))
        var h_ca_inf = 1.0 / (1.0 + exp((v_t + 52.0) / 5.0))
        var m_ih_inf = 1.0 / (1.0 + exp((v_t + 75.0) / 5.5))
        var i_ca_t = self.g_ca_t * m_ca * m_ca * h_ca * (v_t - self.e_ca)
        var i_ih = self.g_ih * m_ih * (v_t - self.e_ih)
        var i_l_t = self.g_l_t * (v_t - self.e_l)
        var i_ts = self.g_st * (v_t - v_s) / self.p_t
        var i_ta = self.g_ta * (v_t - v_a) / self.p_t

        var m_ca_a_inf = 1.0 / (1.0 + exp(-(v_a + 30.0) / 5.0))
        var kca_act = ca_a / (ca_a + 0.001)
        var i_ca_a = self.g_ca_a * m_ca_a_inf * m_ca_a_inf * (v_a - self.e_ca)
        var i_kca = self.g_kca * kca_act * (v_a - self.e_k)
        var i_l_a = self.g_l_a * (v_a - self.e_l)
        var i_at = self.g_ta * (v_a - v_t) / self.p_a

        var d = State(0.0)
        d[0] = (-i_na - i_k - i_l_s - i_st + current_soma / self.p_s) / self.c_m
        d[1] = (h_na_inf - h_na) / tau_h
        d[2] = (n_k_inf - n_k) / tau_n
        d[3] = (-i_ca_t - i_ih - i_l_t - i_ts - i_ta) / self.c_m
        d[4] = m_ca_inf - m_ca
        d[5] = (h_ca_inf - h_ca) / 20.0
        d[6] = (m_ih_inf - m_ih) / 50.0
        d[7] = (-i_ca_a - i_kca - i_l_a - i_at + current_tuft / self.p_a) / self.c_m
        d[8] = -self.f_ca * i_ca_a - ca_a / self.ca_decay
        return d

    fn rk4_substep(self, y: State, current_soma: Float64, current_tuft: Float64) -> State:
        var dt = self.dt
        var k1 = self.derivatives(y, current_soma, current_tuft)
        var k2 = self.derivatives(y + 0.5 * dt * k1, current_soma, current_tuft)
        var k3 = self.derivatives(y + 0.5 * dt * k2, current_soma, current_tuft)
        var k4 = self.derivatives(y + dt * k3, current_soma, current_tuft)
        var next = y + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        next[8] = max(next[8], 0.0)
        return next

    fn simulate(self, n_steps: Int, current_soma: Float64, current_tuft: Float64) -> Int:
        var y = State(0.0)
        y[0] = -75.0
        y[1] = 0.9
        y[2] = 0.1
        y[3] = -75.0
        y[4] = 0.0
        y[5] = 1.0
        y[6] = 0.0
        y[7] = -75.0
        y[8] = 0.0001
        var spikes = 0
        for _ in range(n_steps):
            var v_prev = y[0]
            for _sub in range(N_SUBSTEPS):
                y = self.rk4_substep(y, current_soma, current_tuft)
            if y[0] >= self.v_threshold and v_prev < self.v_threshold:
                spikes += 1
        return spikes


fn main():
    var neuron = HayL5Pyramidal()
    var spikes = neuron.simulate(20000, 10.0, 0.0)
    print("hay_l5 spikes:", spikes)

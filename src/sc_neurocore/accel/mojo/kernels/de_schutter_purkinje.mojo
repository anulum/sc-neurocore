# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo SIMD candidate-first RK4 kernel for de_schutter_purkinje

from math import exp

alias State = SIMD[DType.float64, 8]
alias N_SUBSTEPS = 5


struct DeSchutterPurkinje(Copyable, Movable):
    var g_na: Float64
    var g_k: Float64
    var g_cap: Float64
    var g_kca: Float64
    var g_l: Float64
    var e_na: Float64
    var e_k: Float64
    var e_ca: Float64
    var e_l: Float64
    var ca_decay: Float64
    var f_ca: Float64
    var dt: Float64
    var v_threshold: Float64

    fn __init__(out self):
        self.g_na = 125.0
        self.g_k = 10.0
        self.g_cap = 45.0
        self.g_kca = 35.0
        self.g_l = 0.5
        self.e_na = 45.0
        self.e_k = -85.0
        self.e_ca = 135.0
        self.e_l = -68.0
        self.ca_decay = 0.02
        self.f_ca = 0.00024
        self.dt = 0.01
        self.v_threshold = -20.0

    fn derivatives(self, y: State, current: Float64) -> State:
        var v = y[0]
        var h_na = y[1]
        var n_k = y[2]
        var m_cap = y[3]
        var h_cap = y[4]
        var q_kca = y[5]
        var ca = max(y[6], 0.0)
        var m_na_inf = 1.0 / (1.0 + exp(-(v + 35.0) / 7.5))
        var h_na_inf = 1.0 / (1.0 + exp((v + 55.0) / 7.0))
        var n_k_inf = 1.0 / (1.0 + exp(-(v + 30.0) / 15.0))
        var m_cap_inf = 1.0 / (1.0 + exp(-(v + 19.0) / 5.5))
        var h_cap_inf = 1.0 / (1.0 + exp((v + 48.0) / 7.0))
        var q_kca_inf = ca / (ca + 0.0002)
        var tau_h_na = 0.5 + 14.0 / (1.0 + exp((v + 40.0) / 12.0))
        var tau_n_k = 1.0 + 11.0 / (1.0 + exp((v + 15.0) / 8.0))
        var i_na = self.g_na * m_na_inf * m_na_inf * m_na_inf * h_na * (v - self.e_na)
        var i_k = self.g_k * n_k * n_k * n_k * n_k * (v - self.e_k)
        var i_cap = self.g_cap * m_cap * m_cap * h_cap * (v - self.e_ca)
        var i_kca = self.g_kca * q_kca * (v - self.e_k)
        var i_l = self.g_l * (v - self.e_l)
        var d = State(0.0)
        d[0] = -i_na - i_k - i_cap - i_kca - i_l + current
        d[1] = (h_na_inf - h_na) / tau_h_na
        d[2] = (n_k_inf - n_k) / tau_n_k
        d[3] = (m_cap_inf - m_cap) / 0.3
        d[4] = (h_cap_inf - h_cap) / 45.0
        d[5] = q_kca_inf - q_kca
        d[6] = -self.f_ca * i_cap - self.ca_decay * ca
        return d

    fn rk4_substep(self, y: State, current: Float64) -> State:
        var dt = self.dt
        var k1 = self.derivatives(y, current)
        var k2 = self.derivatives(y + 0.5 * dt * k1, current)
        var k3 = self.derivatives(y + 0.5 * dt * k2, current)
        var k4 = self.derivatives(y + dt * k3, current)
        var next = y + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        next[6] = max(next[6], 0.0)
        return next

    fn simulate(self, n_steps: Int, current: Float64) -> Int:
        var y = State(0.0)
        y[0] = -68.0
        y[1] = 0.8
        y[2] = 0.1
        y[3] = 0.0
        y[4] = 0.9
        y[5] = 0.0
        y[6] = 0.0001
        var spikes = 0
        for _ in range(n_steps):
            var v_prev = y[0]
            for _sub in range(N_SUBSTEPS):
                y = self.rk4_substep(y, current)
            if y[0] >= self.v_threshold and v_prev < self.v_threshold:
                spikes += 1
        return spikes


fn main():
    var neuron = DeSchutterPurkinje()
    var spikes = neuron.simulate(20000, 500.0)
    print("de_schutter_purkinje spikes:", spikes)

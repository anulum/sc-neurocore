# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo SIMD RK4 kernel for the VIP irregular-spiking neuron

from math import exp

# State lanes 0..4 hold (V, h, n, a, b); lanes 5..7 are unused padding so the
# vector width is a power of two for the SIMD RK4 combination.
alias State = SIMD[DType.float64, 8]


struct VIP(Copyable, Movable):
    var g_na: Float64
    var g_k: Float64
    var g_a: Float64
    var g_l: Float64
    var e_na: Float64
    var e_k: Float64
    var e_l: Float64
    var c_m: Float64
    var dt: Float64
    var v_threshold: Float64

    fn __init__(out self, g_a: Float64 = 8.0):
        self.g_na = 35.0
        self.g_k = 6.0
        self.g_a = g_a
        self.g_l = 0.01
        self.e_na = 55.0
        self.e_k = -90.0
        self.e_l = -65.0
        self.c_m = 0.5
        self.dt = 0.025
        self.v_threshold = -20.0

    fn derivatives(self, s: State, current: Float64) -> State:
        var v = s[0]
        var h = s[1]
        var n = s[2]
        var a = s[3]
        var b = s[4]
        var m_inf = 1.0 / (1.0 + exp(-(v + 30.0) / 9.5))
        var h_inf = 1.0 / (1.0 + exp((v + 53.0) / 7.0))
        var tau_h = 0.37 + 2.78 / (1.0 + exp((v + 40.5) / 6.0))
        var n_inf = 1.0 / (1.0 + exp(-(v + 30.0) / 10.0))
        var tau_n = 0.37 + 1.85 / (1.0 + exp((v + 27.0) / 15.0))
        var a_inf = 1.0 / (1.0 + exp(-(v + 50.0) / 20.0))
        var b_inf = 1.0 / (1.0 + exp((v + 78.0) / 6.0))
        var i_na = self.g_na * m_inf * m_inf * m_inf * h * (v - self.e_na)
        var i_k = self.g_k * n * n * n * n * (v - self.e_k)
        var i_a = self.g_a * a * a * a * b * (v - self.e_k)
        var i_l = self.g_l * (v - self.e_l)
        var d = State(0.0)
        d[0] = (-i_na - i_k - i_a - i_l + current) / self.c_m
        d[1] = (h_inf - h) / tau_h
        d[2] = (n_inf - n) / tau_n
        d[3] = (a_inf - a) / 5.0
        d[4] = (b_inf - b) / 50.0
        return d

    fn rk4_substep(self, s: State, current: Float64) -> State:
        var dt = self.dt
        var k1 = self.derivatives(s, current)
        var k2 = self.derivatives(s + 0.5 * dt * k1, current)
        var k3 = self.derivatives(s + 0.5 * dt * k2, current)
        var k4 = self.derivatives(s + dt * k3, current)
        return s + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0

    fn simulate(self, n_steps: Int, current: Float64) -> Int:
        var s = State(0.0)
        s[0] = -65.0
        s[1] = 0.8
        s[2] = 0.1
        s[3] = 0.0
        s[4] = 0.9
        var spikes = 0
        for _ in range(n_steps):
            var v_prev = s[0]
            for _ in range(4):
                s = self.rk4_substep(s, current)
            if s[0] >= self.v_threshold and v_prev < self.v_threshold:
                spikes += 1
        return spikes


fn main():
    var neuron = VIP()
    var spikes = neuron.simulate(40000, 1.0)
    print("vip_neuron spikes:", spikes)

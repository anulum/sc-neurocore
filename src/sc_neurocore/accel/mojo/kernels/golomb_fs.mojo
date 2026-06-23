# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo SIMD RK4 kernel for the Golomb et al. 2007 FS neuron

from math import exp

# State lanes hold (V, h, n, p) — exactly the four-wide SIMD vector.
alias State = SIMD[DType.float64, 4]


struct GolombFS(Copyable, Movable):
    var g_na: Float64
    var g_kd: Float64
    var g_kv3: Float64
    var g_l: Float64
    var e_na: Float64
    var e_k: Float64
    var e_l: Float64
    var c_m: Float64
    var dt: Float64
    var v_threshold: Float64

    fn __init__(out self, g_kv3: Float64 = 150.0):
        self.g_na = 112.5
        self.g_kd = 225.0
        self.g_kv3 = g_kv3
        self.g_l = 0.25
        self.e_na = 50.0
        self.e_k = -90.0
        self.e_l = -70.0
        self.c_m = 1.0
        self.dt = 0.01
        self.v_threshold = -20.0

    fn derivatives(self, s: State, current: Float64) -> State:
        var v = s[0]
        var h = s[1]
        var n = s[2]
        var p = s[3]
        var m_inf = 1.0 / (1.0 + exp(-(v + 24.0) / 11.5))
        var h_inf = 1.0 / (1.0 + exp((v + 58.3) / 6.7))
        var tau_h = 0.5 + 14.0 / (1.0 + exp((v + 60.0) / 12.0))
        var n_inf = 1.0 / (1.0 + exp(-(v + 12.4) / 6.8))
        var tau_n = 0.087 + 11.4 / (1.0 + exp((v + 14.6) / 8.6))
        var p_inf = 1.0 / (1.0 + exp(-(v + 3.0) / 8.0))
        var tau_p = 0.1 + 4.0 / (1.0 + exp((v + 25.0) / 10.0))
        var i_na = self.g_na * m_inf * m_inf * m_inf * h * (v - self.e_na)
        var i_kd = self.g_kd * n * n * n * n * (v - self.e_k)
        var i_kv3 = self.g_kv3 * p * p * (v - self.e_k)
        var i_l = self.g_l * (v - self.e_l)
        var d = State(0.0)
        d[0] = (-i_na - i_kd - i_kv3 - i_l + current) / self.c_m
        d[1] = (h_inf - h) / tau_h
        d[2] = (n_inf - n) / tau_n
        d[3] = (p_inf - p) / tau_p
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
        s[1] = 0.9
        s[2] = 0.1
        s[3] = 0.0
        var spikes = 0
        for _ in range(n_steps):
            var v_prev = s[0]
            for _ in range(10):
                s = self.rk4_substep(s, current)
            if s[0] >= self.v_threshold and v_prev < self.v_threshold:
                spikes += 1
        return spikes


fn main():
    var neuron = GolombFS()
    var spikes = neuron.simulate(40000, 5.0)
    print("golomb_fs FS spikes:", spikes)

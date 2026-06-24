# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo SIMD RK4 kernel for the SST+ low-threshold spiking neuron

from math import exp

# State lanes 0..6 hold (V, m, h, n, p, s, r); lane 7 is unused padding so the
# vector width is a power of two for the SIMD RK4 combination.
alias State = SIMD[DType.float64, 8]


fn asing(num: Float64, slope: Float64, limit: Float64) -> Float64:
    # L'Hôpital limit of the Traub-Miles x/(exp(x/slope)-1) rate at the singularity.
    if abs(num) < 1e-6:
        return limit
    return num / (exp(num / slope) - 1.0)


struct SST(Copyable, Movable):
    var g_na: Float64
    var g_k: Float64
    var g_m: Float64
    var g_t: Float64
    var g_h: Float64
    var g_l: Float64
    var e_na: Float64
    var e_k: Float64
    var e_ca: Float64
    var e_h: Float64
    var e_l: Float64
    var c_m: Float64
    var dt: Float64
    var v_threshold: Float64

    fn __init__(out self, g_t: Float64 = 0.01):
        self.g_na = 50.0
        self.g_k = 5.0
        self.g_m = 0.12
        self.g_t = g_t
        self.g_h = 0.02
        self.g_l = 0.05
        self.e_na = 50.0
        self.e_k = -90.0
        self.e_ca = 120.0
        self.e_h = -40.0
        self.e_l = -65.0
        self.c_m = 1.0
        self.dt = 0.025
        self.v_threshold = -20.0

    fn derivatives(self, y: State, current: Float64) -> State:
        var v = y[0]
        var m = y[1]
        var h = y[2]
        var n = y[3]
        var p = y[4]
        var s = y[5]
        var r = y[6]
        var dvt = v - (-56.2)
        var alpha_m = -0.32 * asing(dvt - 13.0, -4.0, -4.0)
        var beta_m = 0.28 * asing(dvt - 40.0, 5.0, 5.0)
        var alpha_h = 0.128 * exp(-(dvt - 17.0) / 18.0)
        var beta_h = 4.0 / (1.0 + exp(-(dvt - 40.0) / 5.0))
        var alpha_n = -0.032 * asing(dvt - 15.0, -5.0, -5.0)
        var beta_n = 0.5 * exp(-(dvt - 10.0) / 40.0)
        var p_inf = 1.0 / (1.0 + exp(-(v + 35.0) / 10.0))
        var tau_p = 400.0 / (3.3 * exp((v + 35.0) / 20.0) + exp(-(v + 35.0) / 20.0))
        var m_t_inf = 1.0 / (1.0 + exp(-(v + 57.0) / 6.2))
        var s_inf = 1.0 / (1.0 + exp((v + 81.0) / 4.0))
        var tau_s = 30.0 + 200.0 / (1.0 + exp((v + 70.0) / 5.0))
        var r_inf = 1.0 / (1.0 + exp((v + 80.0) / 10.0))
        var tau_r = 100.0 + 500.0 / (exp(-(v + 70.0) / 20.0) + exp((v + 70.0) / 20.0))
        var i_na = self.g_na * m * m * m * h * (v - self.e_na)
        var i_k = self.g_k * n * n * n * n * (v - self.e_k)
        var i_m = self.g_m * p * (v - self.e_k)
        var i_t = self.g_t * m_t_inf * m_t_inf * s * (v - self.e_ca)
        var i_h = self.g_h * r * (v - self.e_h)
        var i_l = self.g_l * (v - self.e_l)
        var d = State(0.0)
        d[0] = (-i_na - i_k - i_m - i_t - i_h - i_l + current) / self.c_m
        d[1] = alpha_m * (1.0 - m) - beta_m * m
        d[2] = alpha_h * (1.0 - h) - beta_h * h
        d[3] = alpha_n * (1.0 - n) - beta_n * n
        d[4] = (p_inf - p) / tau_p
        d[5] = (s_inf - s) / tau_s
        d[6] = (r_inf - r) / tau_r
        return d

    fn rk4_substep(self, y: State, current: Float64) -> State:
        var dt = self.dt
        var k1 = self.derivatives(y, current)
        var k2 = self.derivatives(y + 0.5 * dt * k1, current)
        var k3 = self.derivatives(y + 0.5 * dt * k2, current)
        var k4 = self.derivatives(y + dt * k3, current)
        return y + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0

    fn simulate(self, n_steps: Int, current: Float64) -> Int:
        var y = State(0.0)
        y[0] = -65.0
        y[1] = 0.02
        y[2] = 0.8
        y[3] = 0.2
        y[4] = 0.0
        y[5] = 0.9
        y[6] = 0.1
        var spikes = 0
        for _ in range(n_steps):
            var v_prev = y[0]
            for _ in range(4):
                y = self.rk4_substep(y, current)
            if y[0] >= self.v_threshold and v_prev < self.v_threshold:
                spikes += 1
        return spikes


fn main():
    var neuron = SST()
    var spikes = neuron.simulate(40000, 2.0)
    print("sst_neuron spikes:", spikes)

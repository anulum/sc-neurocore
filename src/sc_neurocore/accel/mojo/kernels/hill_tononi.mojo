# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo SIMD candidate-first RK4 kernel for the Hill-Tononi thalamocortical neuron

from math import exp, sqrt

# State lanes 0..5 hold (V, h_na, n_k, m_h, h_t, na_i); lanes 6..7 are unused
# padding so the vector width is a power of two. One RK4 sub-step of size dt
# advances each call, matching the Python reference's single-step time base.
alias State = SIMD[DType.float64, 8]


struct HillTononi(Copyable, Movable):
    var g_na: Float64
    var g_k: Float64
    var g_h: Float64
    var g_t: Float64
    var g_kna: Float64
    var g_l: Float64
    var e_na: Float64
    var e_k: Float64
    var e_h: Float64
    var e_ca: Float64
    var e_l: Float64
    var na_pump_max: Float64
    var na_eq: Float64
    var dt: Float64
    var v_threshold: Float64

    fn __init__(out self):
        self.g_na = 50.0
        self.g_k = 5.0
        self.g_h = 1.0
        self.g_t = 3.0
        self.g_kna = 1.33
        self.g_l = 0.02
        self.e_na = 50.0
        self.e_k = -90.0
        self.e_h = -43.0
        self.e_ca = 120.0
        self.e_l = -70.0
        self.na_pump_max = 20.0
        self.na_eq = 9.5
        self.dt = 0.05
        self.v_threshold = -20.0

    fn derivatives(self, y: State, current: Float64) -> State:
        var v = y[0]
        var h_na = y[1]
        var n_k = y[2]
        var m_h = y[3]
        var h_t = y[4]
        var na_i = y[5]
        var m_na_inf = 1.0 / (1.0 + exp(-(v + 38.0) / 10.0))
        var h_na_inf = 1.0 / (1.0 + exp((v + 43.0) / 6.0))
        var n_k_inf = 1.0 / (1.0 + exp(-(v + 27.0) / 11.5))
        var m_h_inf = 1.0 / (1.0 + exp((v + 75.0) / 5.5))
        var m_t_inf = 1.0 / (1.0 + exp(-(v + 59.0) / 6.2))
        var h_t_inf = 1.0 / (1.0 + exp((v + 83.0) / 4.0))
        var hill_base = 38.7 / max(na_i, 0.01)
        var w_kna = 0.37 / (1.0 + hill_base * hill_base * hill_base * sqrt(hill_base))
        var tau_h_na = max(1.0 + 10.0 / (1.0 + exp((v + 40.0) / 10.0)), 0.1)
        var z_n = (v + 50.0) / 25.0
        var tau_n_k = max(5.0 + 47.0 * exp(-(z_n * z_n)), 0.1)
        var tau_m_h = max(20.0 + 1000.0 / (exp((v + 71.5) / 14.2) + exp(-(v + 89.0) / 11.6)), 1.0)
        var tau_h_t = 10.0
        if v < -81.0:
            tau_h_t = max(
                30.8 + 211.4 * exp((v + 115.2) / 5.0) / (1.0 + exp((v + 86.0) / 3.2)), 0.1
            )
        var d_h_na = (h_na_inf - h_na) / tau_h_na
        var d_n_k = (n_k_inf - n_k) / tau_n_k
        var d_m_h = (m_h_inf - m_h) / tau_m_h
        var d_h_t = (h_t_inf - h_t) / tau_h_t
        var i_na = self.g_na * m_na_inf * m_na_inf * m_na_inf * h_na * (v - self.e_na)
        var i_k = self.g_k * n_k * n_k * n_k * n_k * (v - self.e_k)
        var i_h = self.g_h * m_h * (v - self.e_h)
        var i_t = self.g_t * m_t_inf * m_t_inf * h_t * (v - self.e_ca)
        var i_kna = self.g_kna * w_kna * (v - self.e_k)
        var i_l = self.g_l * (v - self.e_l)
        var d_v = -i_na - i_k - i_h - i_t - i_kna - i_l + current
        var d_na_i = -0.001 * i_na - self.na_pump_max * (na_i / (na_i + self.na_eq))
        var d = State(0.0)
        d[0] = d_v
        d[1] = d_h_na
        d[2] = d_n_k
        d[3] = d_m_h
        d[4] = d_h_t
        d[5] = d_na_i
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
        y[1] = 0.6
        y[2] = 0.3
        y[3] = 0.0
        y[4] = 0.9
        y[5] = 5.0
        var spikes = 0
        for _ in range(n_steps):
            var v_prev = y[0]
            y = self.rk4_substep(y, current)
            y[5] = max(y[5], 0.0)
            if y[0] >= self.v_threshold and v_prev < self.v_threshold:
                spikes += 1
        return spikes


fn main():
    var neuron = HillTononi()
    var spikes = neuron.simulate(200000, 10.0)
    print("hill_tononi spikes:", spikes)

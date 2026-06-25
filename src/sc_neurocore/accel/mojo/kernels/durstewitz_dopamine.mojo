# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo SIMD candidate-first RK4 kernel for the Durstewitz D1 PFC neuron

from math import exp

# State lanes 0..2 hold (V, h_na, n_k); lane 3 is unused padding so the vector
# width is a power of two. One RK4 sub-step of size dt advances each call, matching
# the Python reference's single-step time base (no inner sub-stepping).
alias State = SIMD[DType.float64, 4]


struct Durstewitz(Copyable, Movable):
    var g_na: Float64
    var g_k: Float64
    var g_nmda: Float64
    var g_l: Float64
    var e_na: Float64
    var e_k: Float64
    var e_nmda: Float64
    var e_l: Float64
    var mg: Float64
    var d1_level: Float64
    var g_nmda_scale: Float64
    var g_k_scale: Float64
    var v_shift_na: Float64
    var dt: Float64
    var v_threshold: Float64

    fn __init__(out self):
        self.g_na = 45.0
        self.g_k = 18.0
        self.g_nmda = 0.5
        self.g_l = 0.02
        self.e_na = 55.0
        self.e_k = -80.0
        self.e_nmda = 0.0
        self.e_l = -65.0
        self.mg = 1.0
        self.d1_level = 0.0
        self.g_nmda_scale = 2.5
        self.g_k_scale = 1.5
        self.v_shift_na = -5.0
        self.dt = 0.05
        self.v_threshold = -20.0

    fn derivatives(self, y: State, current: Float64) -> State:
        var v = y[0]
        var h_na = y[1]
        var n_k = y[2]
        var v_sh = self.d1_level * self.v_shift_na
        var m_na_inf = 1.0 / (1.0 + exp(-(v + 30.0 + v_sh) / 9.5))
        var h_na_inf = 1.0 / (1.0 + exp((v + 53.0) / 7.0))
        var n_k_inf = 1.0 / (1.0 + exp(-(v + 30.0) / 10.0))
        var tau_h = 0.5 + 14.0 / (1.0 + exp((v + 50.0) / 12.0))
        var tau_n = 1.0 + 11.0 / (1.0 + exp((v + 40.0) / 10.0))
        var d_h_na = (h_na_inf - h_na) / tau_h
        var d_n_k = (n_k_inf - n_k) / tau_n
        var mg_block = 1.0 / (1.0 + self.mg / 3.57 * exp(-0.062 * v))
        var nmda_g = self.g_nmda * (1.0 + self.d1_level * (self.g_nmda_scale - 1.0))
        var k_g = self.g_k * (1.0 + self.d1_level * (self.g_k_scale - 1.0))
        var i_na = self.g_na * m_na_inf * m_na_inf * m_na_inf * h_na * (v - self.e_na)
        var i_k = k_g * n_k * n_k * n_k * n_k * (v - self.e_k)
        var i_nmda = nmda_g * mg_block * (v - self.e_nmda)
        var i_l = self.g_l * (v - self.e_l)
        var d_v = -i_na - i_k - i_nmda - i_l + current
        var d = State(0.0)
        d[0] = d_v
        d[1] = d_h_na
        d[2] = d_n_k
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
        y[1] = 0.7
        y[2] = 0.2
        var spikes = 0
        for _ in range(n_steps):
            var v_prev = y[0]
            y = self.rk4_substep(y, current)
            if y[0] >= self.v_threshold and v_prev < self.v_threshold:
                spikes += 1
        return spikes


fn main():
    var neuron = Durstewitz()
    var spikes = neuron.simulate(200000, 10.0)
    print("durstewitz_dopamine spikes:", spikes)

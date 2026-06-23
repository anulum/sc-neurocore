# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo SIMD RK4 kernel for the Pospischil et al. 2008 neuron

from math import exp

# State lanes 0..4 hold (V, m, h, n, p); lanes 5..7 are unused padding so the
# vector width is a power of two for the SIMD RK4 combination.
alias State = SIMD[DType.float64, 8]


@always_inline
fn alpha_singular(num: Float64, slope: Float64, limit: Float64) -> Float64:
    """Traub-Miles rate with the closed-form L'Hôpital limit near the singularity."""
    if abs(num) < 1e-6:
        return limit
    return num / (exp(num / slope) - 1.0)


struct Pospischil(Copyable, Movable):
    var g_na: Float64
    var g_kd: Float64
    var g_m: Float64
    var g_l: Float64
    var e_na: Float64
    var e_k: Float64
    var e_l: Float64
    var c_m: Float64
    var vt: Float64
    var dt: Float64
    var v_threshold: Float64

    fn __init__(out self, g_m: Float64 = 0.07):
        self.g_na = 50.0
        self.g_kd = 5.0
        self.g_m = g_m
        self.g_l = 0.1
        self.e_na = 50.0
        self.e_k = -90.0
        self.e_l = -70.0
        self.c_m = 1.0
        self.vt = -56.2
        self.dt = 0.025
        self.v_threshold = -20.0

    fn derivatives(self, s: State, current: Float64) -> State:
        var v = s[0]
        var m = s[1]
        var h = s[2]
        var n = s[3]
        var p = s[4]
        var dv_vt = v - self.vt
        var am = -0.32 * alpha_singular(dv_vt - 13.0, -4.0, -4.0)
        var bm = 0.28 * alpha_singular(dv_vt - 40.0, 5.0, 5.0)
        var ah = 0.128 * exp(-(dv_vt - 17.0) / 18.0)
        var bh = 4.0 / (1.0 + exp(-(dv_vt - 40.0) / 5.0))
        var an = -0.032 * alpha_singular(dv_vt - 15.0, -5.0, -5.0)
        var bn = 0.5 * exp(-(dv_vt - 10.0) / 40.0)
        var p_inf = 1.0 / (1.0 + exp(-(v + 35.0) / 10.0))
        var tau_p = 608.0 / (3.3 * exp((v + 35.0) / 20.0) + exp(-(v + 35.0) / 20.0))
        var i_na = self.g_na * m * m * m * h * (v - self.e_na)
        var i_kd = self.g_kd * n * n * n * n * (v - self.e_k)
        var i_m = self.g_m * p * (v - self.e_k)
        var i_l = self.g_l * (v - self.e_l)
        var d = State(0.0)
        d[0] = (-i_na - i_kd - i_m - i_l + current) / self.c_m
        d[1] = am * (1.0 - m) - bm * m
        d[2] = ah * (1.0 - h) - bh * h
        d[3] = an * (1.0 - n) - bn * n
        d[4] = (p_inf - p) / tau_p
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
        s[0] = -70.0
        s[1] = 0.05
        s[2] = 0.6
        s[3] = 0.3
        s[4] = 0.0
        var spikes = 0
        for _ in range(n_steps):
            var v_prev = s[0]
            for _ in range(4):
                s = self.rk4_substep(s, current)
            if s[0] >= self.v_threshold and v_prev < self.v_threshold:
                spikes += 1
        return spikes


fn main():
    var neuron = Pospischil()
    var spikes = neuron.simulate(40000, 7.0)
    print("pospischil RS spikes:", spikes)

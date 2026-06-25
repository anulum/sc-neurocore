# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo exponential-Euler kernel for the upper motor (L5 corticospinal) neuron

from math import exp

# State lanes 0..5 hold (V, m, h, n, p, s); lanes 6..7 are unused padding so the
# vector width is a power of two. The membrane uses an analytic exponential-Euler
# step (frozen-conductance) and the gates use the closed-form steady/tau update,
# both of which are unconditionally stable for the stiff sodium gate — RK4 here
# would be a regression, not a hardening.
alias State = SIMD[DType.float64, 8]


fn rate_exp(value: Float64) -> Float64:
    return exp(max(-60.0, min(60.0, value)))


fn gate(previous: Float64, alpha: Float64, beta: Float64, dt: Float64) -> Float64:
    # Closed-form Hodgkin-Huxley gate step: relax towards alpha/(alpha+beta)
    # with the exact exponential of the combined rate over the timestep.
    var total = alpha + beta
    var steady = alpha / total
    return min(1.0, max(0.0, steady + (previous - steady) * rate_exp(-total * dt)))


fn gate_inf(previous: Float64, steady: Float64, tau: Float64, dt: Float64) -> Float64:
    # Closed-form steady-state/tau gate step for the M-current and Ca2+ activation.
    return min(1.0, max(0.0, steady + (previous - steady) * rate_exp(-dt / tau)))


struct UpperMotor(Copyable, Movable):
    var g_na: Float64
    var g_k: Float64
    var g_m: Float64
    var g_ca: Float64
    var g_l: Float64
    var e_na: Float64
    var e_k: Float64
    var e_ca: Float64
    var e_l: Float64
    var c_m: Float64
    var dt: Float64
    var v_threshold: Float64

    fn __init__(out self):
        self.g_na = 50.0
        self.g_k = 5.0
        self.g_m = 0.07
        self.g_ca = 0.3
        self.g_l = 0.1
        self.e_na = 50.0
        self.e_k = -90.0
        self.e_ca = 120.0
        self.e_l = -70.0
        self.c_m = 1.0
        self.dt = 0.025
        self.v_threshold = -20.0

    fn step_candidate(self, y: State, current: Float64) -> State:
        var v = y[0]
        var m = y[1]
        var h = y[2]
        var n = y[3]
        var p = y[4]
        var s = y[5]
        var dv = v - (-56.2)
        var x_m = dv - 13.0
        var alpha_m = -0.32 * x_m / (rate_exp(-x_m / 4.0) - 1.0)
        if abs(x_m) < 1e-6:
            alpha_m = 0.32 * 4.0
        # Published Pospischil beta_m numerator is V - V_T - 40; the earlier -17
        # offset (shared with alpha_h) drove the cell into depolarisation block.
        var x_bm = dv - 40.0
        var beta_m = 0.28 * x_bm / (rate_exp(x_bm / 5.0) - 1.0)
        if abs(x_bm) < 1e-6:
            beta_m = 0.28 * 5.0
        var alpha_h = 0.128 * rate_exp(-(dv - 17.0) / 18.0)
        var beta_h = 4.0 / (1.0 + rate_exp(-(dv - 40.0) / 5.0))
        var x_n = dv - 15.0
        var alpha_n = -0.032 * x_n / (rate_exp(-x_n / 5.0) - 1.0)
        if abs(x_n) < 1e-6:
            alpha_n = 0.032 * 5.0
        var beta_n = 0.5 * rate_exp(-(dv - 10.0) / 40.0)
        var next_m = gate(m, alpha_m, beta_m, self.dt)
        var next_h = gate(h, alpha_h, beta_h, self.dt)
        var next_n = gate(n, alpha_n, beta_n, self.dt)
        var p_inf = 1.0 / (1.0 + rate_exp(-(v + 35.0) / 10.0))
        var tau_p = 400.0 / (3.3 * rate_exp((v + 35.0) / 20.0) + rate_exp(-(v + 35.0) / 20.0))
        var next_p = gate_inf(p, p_inf, tau_p, self.dt)
        var s_inf = 1.0 / (1.0 + rate_exp(-(v + 20.0) / 5.0))
        var next_s = gate_inf(s, s_inf, 10.0, self.dt)
        var g_na = self.g_na * next_m * next_m * next_m * next_h
        var g_k = self.g_k * next_n * next_n * next_n * next_n
        var g_m = self.g_m * next_p
        var g_ca = self.g_ca * next_s * next_s
        var g_total = g_na + g_k + g_m + g_ca + self.g_l
        var steady_v = (
            current
            + g_na * self.e_na
            + g_k * self.e_k
            + g_m * self.e_k
            + g_ca * self.e_ca
            + self.g_l * self.e_l
        ) / g_total
        var next_v = steady_v + (v - steady_v) * rate_exp(-(g_total / self.c_m) * self.dt)
        var out = State(0.0)
        out[0] = max(-150.0, min(100.0, next_v))
        out[1] = next_m
        out[2] = next_h
        out[3] = next_n
        out[4] = next_p
        out[5] = next_s
        return out

    fn simulate(self, n_steps: Int, current: Float64) -> Int:
        var y = State(0.0)
        y[0] = -70.0
        y[1] = 0.05
        y[2] = 0.6
        y[3] = 0.3
        y[4] = 0.0
        y[5] = 0.0
        var spikes = 0
        for _ in range(n_steps):
            var v_prev = y[0]
            for _ in range(4):
                y = self.step_candidate(y, current)
            if y[0] >= self.v_threshold and v_prev < self.v_threshold:
                spikes += 1
        return spikes


fn main():
    var neuron = UpperMotor()
    var spikes = neuron.simulate(40000, 5.0)
    print("upper_motor_neuron spikes:", spikes)

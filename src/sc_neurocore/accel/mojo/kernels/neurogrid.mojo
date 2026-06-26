# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo SIMD candidate-first RK4 kernel for neurogrid

from math import exp

alias State = SIMD[DType.float64, 4]


struct NeuroGrid(Copyable, Movable):
    var tau_s: Float64
    var tau_d: Float64
    var g_c: Float64
    var delta_t: Float64
    var v_rest: Float64
    var v_threshold: Float64
    var v_peak: Float64
    var v_reset: Float64
    var dt: Float64

    fn __init__(out self):
        self.tau_s = 20.0
        self.tau_d = 50.0
        self.g_c = 0.5
        self.delta_t = 2.0
        self.v_rest = -65.0
        self.v_threshold = -50.0
        self.v_peak = 20.0
        self.v_reset = -65.0
        self.dt = 0.1

    fn derivatives(self, y: State, current: Float64) -> State:
        var v_s_eff = min(y[0], self.v_peak)
        var v_d = y[1]
        var exp_arg = min((v_s_eff - self.v_threshold) / self.delta_t, 20.0)
        var exp_term = self.delta_t * exp(exp_arg)
        var d = State(0.0)
        d[0] = (-(v_s_eff - self.v_rest) + exp_term + self.g_c * (v_d - v_s_eff)) / self.tau_s
        d[1] = (-(v_d - self.v_rest) + current - self.g_c * (v_d - v_s_eff)) / self.tau_d
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
        y[1] = -65.0
        var spikes = 0
        for _ in range(n_steps):
            y = self.rk4_substep(y, current)
            if y[0] >= self.v_peak:
                y[0] = self.v_reset
                spikes += 1
        return spikes


fn main():
    var neuron = NeuroGrid()
    var spikes = neuron.simulate(20000, 100.0)
    print("neurogrid spikes:", spikes)

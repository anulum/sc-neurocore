# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo SIMD candidate-first RK4 kernel for the multicompartment MCN

from math import exp

# State lanes 0..2 hold (u, v_basal, v_apical); lane 3 is unused padding so the
# vector width is a power of two. One RK4 sub-step of size dt advances each call.
alias State = SIMD[DType.float64, 4]


struct MulticompartmentMCN(Copyable, Movable):
    var tau: Float64
    var tau_b: Float64
    var tau_a: Float64
    var g_ratio: Float64
    var beta: Float64
    var v_th: Float64
    var dt: Float64

    fn __init__(out self):
        self.tau = 2.0
        self.tau_b = 2.0
        self.tau_a = 2.0
        self.g_ratio = 1.0
        self.beta = 1.0
        self.v_th = 1.0
        self.dt = 1.0

    fn sigma(self, x: Float64) -> Float64:
        return 1.0 / (1.0 + exp(-self.beta * x))

    fn derivatives(self, y: State, x_basal: Float64, x_apical: Float64, i_soma: Float64) -> State:
        var u = y[0]
        var v_basal = y[1]
        var v_apical = y[2]
        var gate = self.sigma(v_apical)
        var d = State(0.0)
        d[0] = (-u + gate * (self.g_ratio * (v_basal - u) + i_soma)) / self.tau
        d[1] = (-v_basal + x_basal) / self.tau_b
        d[2] = (-v_apical + x_apical) / self.tau_a
        return d

    fn rk4_substep(self, y: State, x_basal: Float64, x_apical: Float64, i_soma: Float64) -> State:
        var dt = self.dt
        var k1 = self.derivatives(y, x_basal, x_apical, i_soma)
        var k2 = self.derivatives(y + 0.5 * dt * k1, x_basal, x_apical, i_soma)
        var k3 = self.derivatives(y + 0.5 * dt * k2, x_basal, x_apical, i_soma)
        var k4 = self.derivatives(y + dt * k3, x_basal, x_apical, i_soma)
        return y + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0

    fn simulate(self, n_steps: Int, current: Float64) -> Int:
        var y = State(0.0)
        var spikes = 0
        for _ in range(n_steps):
            y = self.rk4_substep(y, current, 0.0, 0.0)
            if y[0] >= self.v_th:
                y[0] = 0.0
                spikes += 1
        return spikes


fn main():
    var neuron = MulticompartmentMCN()
    var spikes = neuron.simulate(200000, 3.2)
    print("multicompartment_mcn spikes:", spikes)

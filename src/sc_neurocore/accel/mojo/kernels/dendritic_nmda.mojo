# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo candidate-first RK4 kernel for dendritic NMDA

from math import exp

alias State = SIMD[DType.float64, 4]


struct DendriticNMDA(Copyable, Movable):
    var g_nmda: Float64
    var e_nmda: Float64
    var mg_conc: Float64
    var g_coupling: Float64
    var tau_soma: Float64
    var tau_dend: Float64
    var theta: Float64
    var dt: Float64

    fn __init__(out self):
        self.g_nmda = 1.5
        self.e_nmda = 0.0
        self.mg_conc = 1.0
        self.g_coupling = 0.5
        self.tau_soma = 20.0
        self.tau_dend = 50.0
        self.theta = -50.0
        self.dt = 0.1

    fn mg_block(self, v_dend: Float64) -> Float64:
        return 1.0 / (1.0 + (self.mg_conc / 3.57) * exp(-0.062 * v_dend))

    fn derivatives(self, y: State, i_soma: Float64, glutamate: Float64) -> State:
        var v_soma = y[0]
        var v_dend = y[1]
        var block = self.mg_block(v_dend)
        var i_nmda = self.g_nmda * glutamate * block * (v_dend - self.e_nmda)
        var d = State(0.0)
        d[0] = (-v_soma - 65.0 + i_soma + self.g_coupling * (v_dend - v_soma)) / self.tau_soma
        d[1] = (-v_dend - 65.0 + i_nmda + self.g_coupling * (v_soma - v_dend)) / self.tau_dend
        return d

    fn rk4_substep(self, y: State, i_soma: Float64, glutamate: Float64) -> State:
        var dt = self.dt
        var k1 = self.derivatives(y, i_soma, glutamate)
        var k2 = self.derivatives(y + 0.5 * dt * k1, i_soma, glutamate)
        var k3 = self.derivatives(y + 0.5 * dt * k2, i_soma, glutamate)
        var k4 = self.derivatives(y + dt * k3, i_soma, glutamate)
        return y + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0

    fn simulate(self, n_steps: Int, i_soma: Float64, glutamate: Float64) -> Int:
        var y = State(0.0)
        y[0] = -65.0
        y[1] = -65.0
        var spikes = 0
        for _ in range(n_steps):
            y = self.rk4_substep(y, i_soma, glutamate)
            if y[0] >= self.theta:
                y[0] = -65.0
                spikes += 1
        return spikes


fn main():
    var neuron = DendriticNMDA()
    var spikes = neuron.simulate(20000, 50.0, 0.5)
    print("dendritic_nmda spikes:", spikes)

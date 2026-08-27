# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo Wang 1999 NMDA-autapse neuron

from std.math import exp, isfinite

comptime State = SIMD[DType.float64, 4]


struct NMDANeuron(Copyable, Movable):
    var v: Float64
    var x_nmda: Float64
    var s_nmda: Float64
    var ca: Float64
    var refractory_remaining: Float64
    var c_m: Float64
    var g_l: Float64
    var v_l: Float64
    var g_nmda: Float64
    var e_nmda: Float64
    var mg_conc: Float64
    var alpha_x: Float64
    var tau_x: Float64
    var alpha_s: Float64
    var tau_s: Float64
    var kinetic_scale: Float64
    var g_ahp: Float64
    var v_k: Float64
    var alpha_ca: Float64
    var tau_ca: Float64
    var dt: Float64
    var v_threshold: Float64
    var v_reset: Float64
    var refractory_period: Float64

    def __init__(out self):
        self.v, self.x_nmda, self.s_nmda, self.ca = -70.0, 0.0, 0.0, 0.0
        self.refractory_remaining = 0.0
        self.c_m, self.g_l, self.v_l = 0.5, 0.025, -70.0
        self.g_nmda, self.e_nmda, self.mg_conc = 0.1, 0.0, 1.0
        self.alpha_x, self.tau_x = 1.0, 2.0
        self.alpha_s, self.tau_s, self.kinetic_scale = 1.0, 80.0, 1.0
        self.g_ahp, self.v_k, self.alpha_ca, self.tau_ca = 0.0, -85.0, 0.2, 80.0
        self.dt, self.v_threshold, self.v_reset = 0.05, -52.0, -59.0
        self.refractory_period = 2.0

    def valid(self) -> Bool:
        return (
            isfinite(self.v)
            and self.v >= -120.0
            and self.v <= 80.0
            and isfinite(self.x_nmda)
            and self.x_nmda >= 0.0
            and isfinite(self.s_nmda)
            and self.s_nmda >= 0.0
            and self.s_nmda <= 1.0
            and isfinite(self.ca)
            and self.ca >= 0.0
            and isfinite(self.refractory_remaining)
            and self.refractory_remaining >= 0.0
            and self.refractory_remaining <= self.refractory_period
            and isfinite(self.c_m)
            and self.c_m >= 0.01
            and self.c_m <= 10.0
            and isfinite(self.g_l)
            and self.g_l >= 0.0
            and self.g_l <= 1.0
            and isfinite(self.v_l)
            and self.v_l >= -100.0
            and self.v_l <= -40.0
            and isfinite(self.g_nmda)
            and self.g_nmda >= 0.0
            and self.g_nmda <= 2.0
            and isfinite(self.e_nmda)
            and self.e_nmda >= -10.0
            and self.e_nmda <= 10.0
            and isfinite(self.mg_conc)
            and self.mg_conc >= 0.0
            and self.mg_conc <= 5.0
            and isfinite(self.alpha_x)
            and self.alpha_x >= 0.0
            and self.alpha_x <= 10.0
            and isfinite(self.tau_x)
            and self.tau_x >= 0.01
            and self.tau_x <= 100.0
            and isfinite(self.alpha_s)
            and self.alpha_s >= 0.0
            and self.alpha_s <= 10.0
            and isfinite(self.tau_s)
            and self.tau_s >= 1.0
            and self.tau_s <= 1000.0
            and isfinite(self.kinetic_scale)
            and self.kinetic_scale >= 0.01
            and self.kinetic_scale <= 100.0
            and isfinite(self.g_ahp)
            and self.g_ahp >= 0.0
            and self.g_ahp <= 10.0
            and isfinite(self.v_k)
            and self.v_k >= -120.0
            and self.v_k <= -40.0
            and isfinite(self.alpha_ca)
            and self.alpha_ca >= 0.0
            and self.alpha_ca <= 10.0
            and isfinite(self.tau_ca)
            and self.tau_ca >= 1.0
            and self.tau_ca <= 1000.0
            and isfinite(self.dt)
            and self.dt > 0.0
            and self.dt <= 0.05
            and isfinite(self.v_threshold)
            and self.v_threshold >= -80.0
            and self.v_threshold <= -30.0
            and isfinite(self.v_reset)
            and self.v_reset >= -100.0
            and self.v_reset < self.v_threshold
            and isfinite(self.refractory_period)
            and self.refractory_period >= 0.0
            and self.refractory_period <= 20.0
        )

    def derivatives(self, state: State, current: Float64) -> State:
        var v, x, gate, calcium = state[0], state[1], state[2], state[3]
        var block = 1.0 / (1.0 + self.mg_conc * exp(-0.062 * v) / 3.57)
        var i_l = self.g_l * (v - self.v_l)
        var i_ahp = self.g_ahp * calcium * (v - self.v_k)
        var i_nmda = self.g_nmda * gate * block * (v - self.e_nmda)
        var result = State(0.0)
        result[0] = (-i_l - i_ahp - i_nmda + current) / self.c_m
        result[1] = self.kinetic_scale * (-x / self.tau_x)
        result[2] = self.kinetic_scale * (
            self.alpha_s * x * (1.0 - gate) - gate / self.tau_s
        )
        result[3] = -calcium / self.tau_ca
        return result

    def step(mut self, current: Float64) raises -> Int:
        if not isfinite(current) or not self.valid():
            raise Error("invalid NMDA state, configuration, or input")
        var held = self.refractory_remaining > 0.0
        var state = State(0.0)
        state[0] = self.v_reset if held else self.v
        state[1], state[2], state[3] = self.x_nmda, self.s_nmda, self.ca
        var k1 = self.derivatives(state, current)
        var k2 = self.derivatives(state + 0.5 * self.dt * k1, current)
        var candidate = state + self.dt * k2
        var refractory = max(0.0, self.refractory_remaining - self.dt)
        var event = 0
        if held:
            candidate[0] = self.v_reset
        elif candidate[0] >= self.v_threshold:
            event = 1
            candidate[0] = self.v_reset
            refractory = self.refractory_period
            candidate[1] += self.kinetic_scale * self.alpha_x
            candidate[3] += self.alpha_ca
        for index in range(4):
            if not isfinite(candidate[index]):
                raise Error("NMDA RK2 candidate must be finite")
        self.v = max(-120.0, min(80.0, candidate[0]))
        self.x_nmda = max(0.0, candidate[1])
        self.s_nmda = max(0.0, min(1.0, candidate[2]))
        self.ca = max(0.0, candidate[3])
        self.refractory_remaining = refractory
        return event


def main() raises:
    var anchor = NMDANeuron()
    var event = anchor.step(0.3)
    print(
        event,
        anchor.v,
        anchor.x_nmda,
        anchor.s_nmda,
        anchor.ca,
        anchor.refractory_remaining,
    )
    var trajectory = NMDANeuron()
    var events = 0
    for _ in range(10000):
        events += trajectory.step(0.6)
    print(events, trajectory.v, trajectory.x_nmda, trajectory.s_nmda, trajectory.ca)

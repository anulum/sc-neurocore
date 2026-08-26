# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo Butera-Rinzel-Smith 1999 Model 1

from std.math import cosh, exp, isfinite

comptime State = SIMD[DType.float64, 4]


# Three-state Model 1 with the repository's explicit RK4 specialization.
struct ButeraRespiratory(Copyable, Movable):
    var v: Float64
    var n: Float64
    var h_nap: Float64
    var g_na: Float64
    var g_nap: Float64
    var g_k: Float64
    var g_l: Float64
    var capacitance: Float64
    var e_na: Float64
    var e_k: Float64
    var e_l: Float64
    var g_tonic: Float64
    var e_syn: Float64
    var tau_h: Float64
    var dt: Float64
    var v_threshold: Float64

    def __init__(out self):
        self.v = -50.0
        self.n = 0.01
        self.h_nap = 0.5
        self.g_na = 28.0
        self.g_nap = 2.8
        self.g_k = 11.2
        self.g_l = 2.8
        self.capacitance = 21.0
        self.e_na = 50.0
        self.e_k = -85.0
        self.e_l = -65.0
        self.g_tonic = 0.0
        self.e_syn = 0.0
        self.tau_h = 10000.0
        self.dt = 0.1
        self.v_threshold = -20.0

    def configuration_is_valid(self) -> Bool:
        return (
            isfinite(self.v) and self.v >= -200.0 and self.v <= 100.0
            and isfinite(self.n) and self.n >= -0.05 and self.n <= 1.05
            and isfinite(self.h_nap) and self.h_nap >= -0.05
            and self.h_nap <= 1.05 and isfinite(self.g_na) and self.g_na >= 0.0
            and isfinite(self.g_nap) and self.g_nap >= 0.0
            and isfinite(self.g_k) and self.g_k >= 0.0
            and isfinite(self.g_l) and self.g_l >= 0.0
            and isfinite(self.capacitance) and self.capacitance > 0.0
            and isfinite(self.e_na) and isfinite(self.e_k) and isfinite(self.e_l)
            and isfinite(self.g_tonic) and self.g_tonic >= 0.0
            and isfinite(self.e_syn) and isfinite(self.tau_h) and self.tau_h > 0.0
            and isfinite(self.dt) and self.dt > 0.0
            and isfinite(self.v_threshold)
        )

    def derivatives(self, state: State, current: Float64) -> State:
        var v = min(100.0, max(-200.0, state[0]))
        var n = min(1.0, max(0.0, state[1]))
        var h_nap = min(1.0, max(0.0, state[2]))
        var m_na = 1.0 / (1.0 + exp(-(v + 34.0) / 5.0))
        var m_nap = 1.0 / (1.0 + exp(-(v + 40.0) / 6.0))
        var h_inf = 1.0 / (1.0 + exp((v + 48.0) / 6.0))
        var n_inf = 1.0 / (1.0 + exp(-(v + 29.0) / 4.0))
        var tau_n = max(10.0 / max(cosh((v + 29.0) / 8.0), 1e-12), 0.01)
        var tau_h = max(self.tau_h / max(cosh((v + 48.0) / 12.0), 1e-12), 0.1)
        var i_na = self.g_na * m_na * m_na * m_na * (1.0 - n) * (v - self.e_na)
        var i_nap = self.g_nap * m_nap * h_nap * (v - self.e_na)
        var i_k = self.g_k * n * n * n * n * (v - self.e_k)
        var i_l = self.g_l * (v - self.e_l)
        var i_tonic = self.g_tonic * (v - self.e_syn)
        var result = State(0.0)
        result[0] = (
            -i_na - i_nap - i_k - i_l - i_tonic + current
        ) / self.capacitance
        result[1] = (n_inf - n) / tau_n
        result[2] = (h_inf - h_nap) / tau_h
        return result

    def rk4_candidate(self, state: State, current: Float64) -> State:
        var k1 = self.derivatives(state, current)
        var k2 = self.derivatives(state + 0.5 * self.dt * k1, current)
        var k3 = self.derivatives(state + 0.5 * self.dt * k2, current)
        var k4 = self.derivatives(state + self.dt * k3, current)
        return state + self.dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0

    # Advance atomically and return an observational upward-crossing event.
    def step(mut self, current: Float64) raises -> Int:
        if not isfinite(current) or not self.configuration_is_valid():
            raise Error("Butera Model 1 state and input must be finite and physical")
        var state = State(0.0)
        state[0] = self.v
        state[1] = self.n
        state[2] = self.h_nap
        var candidate = self.rk4_candidate(state, current)
        for index in range(3):
            if not isfinite(candidate[index]):
                raise Error("Butera Model 1 RK4 candidate must be finite")
        var v_previous = self.v
        self.v = min(100.0, max(-200.0, candidate[0]))
        self.n = min(1.0, max(0.0, candidate[1]))
        self.h_nap = min(1.0, max(0.0, candidate[2]))
        return 1 if self.v >= self.v_threshold and v_previous < self.v_threshold else 0

    def simulate(mut self, n_steps: Int, current: Float64) raises -> Int:
        var spikes = 0
        for _ in range(n_steps):
            spikes += self.step(current)
        return spikes


def main() raises:
    var anchor = ButeraRespiratory()
    var event = anchor.step(12.5)
    print(anchor.v, anchor.n, anchor.h_nap, event)
    var neuron = ButeraRespiratory()
    var spikes = neuron.simulate(20000, 50.0)
    print("butera_model1 spikes:", spikes)
    if spikes != 173:
        raise Error("Butera Model 1 source parity failed")

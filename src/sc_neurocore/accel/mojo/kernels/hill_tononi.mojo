# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo Hill-Tononi 2005 hybrid neuron

from std.math import exp, isfinite, log, sqrt

alias State = SIMD[DType.float64, 8]


# Hill-Tononi cortical-waking state and complete scalar-cell configuration.
struct HillTononi(Copyable, Movable):
    var v: Float64
    var theta: Float64
    var d_k: Float64
    var m_h: Float64
    var m_t: Float64
    var h_t: Float64
    var spike_timer: Float64
    var g_na_l: Float64
    var g_k_l: Float64
    var g_na_p: Float64
    var g_dk: Float64
    var g_h: Float64
    var g_t: Float64
    var e_na: Float64
    var e_k: Float64
    var e_na_p: Float64
    var e_dk: Float64
    var e_h: Float64
    var e_t: Float64
    var n_na_p: Float64
    var n_t: Float64
    var tau_m: Float64
    var theta_eq: Float64
    var tau_theta: Float64
    var g_spike: Float64
    var t_spike: Float64
    var tau_spike: Float64
    var tau_d: Float64
    var d_influx_peak: Float64
    var d_threshold: Float64
    var d_slope: Float64
    var d_eq: Float64
    var d_half: Float64
    var dt: Float64

    # Return the publication's cortical-excitatory waking profile.
    def __init__(out self):
        self.v = -70.0
        self.theta = -51.0
        self.d_k = 0.001
        self.m_h = 0.2871859013825026
        self.m_t = 0.1450215950687922
        self.h_t = 0.03732688734412946
        self.spike_timer = 0.0
        self.g_na_l = 0.2
        self.g_k_l = 1.0
        self.g_na_p = 0.5
        self.g_dk = 0.5
        self.g_h = 0.0
        self.g_t = 0.0
        self.e_na = 30.0
        self.e_k = -90.0
        self.e_na_p = 30.0
        self.e_dk = -90.0
        self.e_h = -40.0
        self.e_t = 0.0
        self.n_na_p = 3.0
        self.n_t = 2.0
        self.tau_m = 16.0
        self.theta_eq = -51.0
        self.tau_theta = 2.0
        self.g_spike = 1.0
        self.t_spike = 2.0
        self.tau_spike = 1.75
        self.tau_d = 1250.0
        self.d_influx_peak = 0.025
        self.d_threshold = -10.0
        self.d_slope = 5.0
        self.d_eq = 0.001
        self.d_half = 0.25
        self.dt = 0.25

    fn configuration_is_valid(self) -> Bool:
        return (
            isfinite(self.v) and isfinite(self.theta) and isfinite(self.d_k)
            and isfinite(self.m_h) and isfinite(self.m_t) and isfinite(self.h_t)
            and isfinite(self.spike_timer) and self.d_k >= 0.0
            and self.spike_timer >= 0.0 and isfinite(self.g_na_l)
            and self.g_na_l >= 0.0 and isfinite(self.g_k_l)
            and self.g_k_l >= 0.0 and isfinite(self.g_na_p)
            and self.g_na_p >= 0.0 and isfinite(self.g_dk)
            and self.g_dk >= 0.0 and isfinite(self.g_h) and self.g_h >= 0.0
            and isfinite(self.g_t) and self.g_t >= 0.0
            and isfinite(self.e_na) and isfinite(self.e_k)
            and isfinite(self.e_na_p) and isfinite(self.e_dk)
            and isfinite(self.e_h) and isfinite(self.e_t)
            and isfinite(self.n_na_p) and self.n_na_p > 0.0
            and isfinite(self.n_t) and self.n_t > 0.0
            and isfinite(self.tau_m) and self.tau_m > 0.0
            and isfinite(self.theta_eq) and isfinite(self.tau_theta)
            and self.tau_theta > 0.0 and isfinite(self.g_spike)
            and self.g_spike >= 0.0 and isfinite(self.t_spike)
            and self.t_spike > 0.0 and isfinite(self.tau_spike)
            and self.tau_spike > 0.0 and isfinite(self.tau_d)
            and self.tau_d > 0.0 and isfinite(self.d_influx_peak)
            and self.d_influx_peak >= 0.0 and isfinite(self.d_threshold)
            and isfinite(self.d_slope) and self.d_slope > 0.0
            and isfinite(self.d_eq) and self.d_eq >= 0.0
            and isfinite(self.d_half) and self.d_half > 0.0
            and isfinite(self.dt) and self.dt > 0.0
        )

    fn derivatives(self, y: State, current: Float64, spike_active: Bool) -> State:
        var v = y[0]
        var theta = y[1]
        var d_k = y[2]
        var m_h = y[3]
        var m_t = y[4]
        var h_t = y[5]
        var m_na_p = 1.0 / (1.0 + exp(-(v + 55.7) / 7.7))
        var d_base = self.d_half / max(d_k, 1e-15)
        var d_activation = 1.0 / (1.0 + d_base * d_base * d_base * sqrt(d_base))
        var d_influx = self.d_influx_peak / (
            1.0 + exp(-(v - self.d_threshold) / self.d_slope)
        )
        var d_k_inf = self.tau_d * d_influx + self.d_eq
        var m_h_inf = 1.0 / (1.0 + exp((v + 75.0) / 5.5))
        var tau_m_h = 1.0 / (
            exp(-14.59 - 0.086 * v) + exp(-1.87 + 0.0701 * v)
        )
        var m_t_inf = 1.0 / (1.0 + exp(-(v + 59.0) / 6.2))
        var tau_m_t = 0.22 / (
            exp(-(v + 132.0) / 16.7) + exp((v + 16.8) / 18.2)
        ) + 0.13
        var h_t_inf = 1.0 / (1.0 + exp((v + 83.0) / 4.0))
        var tau_h_t = 8.2 + (
            56.6 + 0.27 * exp((v + 115.2) / 5.0)
        ) / (1.0 + exp((v + 86.0) / 3.2))
        var i_na_l = -self.g_na_l * (v - self.e_na)
        var i_k_l = -self.g_k_l * (v - self.e_k)
        var i_na_p = -self.g_na_p * exp(self.n_na_p * log(m_na_p)) * (
            v - self.e_na_p
        )
        var i_dk = -self.g_dk * d_activation * (v - self.e_dk)
        var i_h = -self.g_h * m_h * (v - self.e_h)
        var i_t = -self.g_t * exp(self.n_t * log(m_t)) * h_t * (v - self.e_t)
        var i_spike = 0.0
        if spike_active:
            i_spike = -self.g_spike * (v - self.e_k) / self.tau_spike
        var derivative = State(0.0)
        derivative[0] = (
            i_na_l + i_k_l + i_na_p + i_dk + i_h + i_t + current
        ) / self.tau_m + i_spike
        derivative[1] = -(theta - self.theta_eq) / self.tau_theta
        derivative[2] = (d_k_inf - d_k) / self.tau_d
        derivative[3] = (m_h_inf - m_h) / tau_m_h
        derivative[4] = (m_t_inf - m_t) / tau_m_t
        derivative[5] = (h_t_inf - h_t) / tau_h_t
        return derivative

    fn rk4_candidate(self, y: State, current: Float64, spike_active: Bool) -> State:
        var k1 = self.derivatives(y, current, spike_active)
        var k2 = self.derivatives(y + 0.5 * self.dt * k1, current, spike_active)
        var k3 = self.derivatives(y + 0.5 * self.dt * k2, current, spike_active)
        var k4 = self.derivatives(y + self.dt * k3, current, spike_active)
        return y + self.dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0

    # Advance one source RK4 step and commit state only after validation.
    def step(mut self, current: Float64) raises -> Int:
        if not isfinite(current) or not self.configuration_is_valid():
            raise Error("Hill-Tononi state and current must be finite and physical")
        var refractory = self.spike_timer > 0.0
        var state = State(0.0)
        state[0] = self.v
        state[1] = self.theta
        state[2] = self.d_k
        state[3] = self.m_h
        state[4] = self.m_t
        state[5] = self.h_t
        var candidate = self.rk4_candidate(state, current, refractory)
        for index in range(6):
            if not isfinite(candidate[index]):
                raise Error("Hill-Tononi candidate must be finite")
        if candidate[2] < 0.0:
            raise Error("Hill-Tononi D candidate must be non-negative")
        var timer = max(0.0, self.spike_timer - self.dt)
        var spike = not refractory and candidate[0] >= candidate[1]
        if spike:
            candidate[0] = self.e_na
            candidate[1] = self.e_na
            timer = self.t_spike
        self.v = candidate[0]
        self.theta = candidate[1]
        self.d_k = candidate[2]
        self.m_h = candidate[3]
        self.m_t = candidate[4]
        self.h_t = candidate[5]
        self.spike_timer = timer
        return 1 if spike else 0

    # Simulate at constant external current and return the event count.
    def simulate(mut self, n_steps: Int, current: Float64) raises -> Int:
        var spikes = 0
        for _ in range(n_steps):
            spikes += self.step(current)
        return spikes


def main() raises:
    var anchor = HillTononi()
    var event = anchor.step(12.0)
    print(anchor.v, anchor.theta, anchor.d_k, anchor.m_h, anchor.m_t, anchor.h_t, event)
    var neuron = HillTononi()
    var spikes = neuron.simulate(200000, 20.0)
    print("hill_tononi spikes:", spikes)
    if spikes != 538:
        raise Error("Hill-Tononi source parity failed")

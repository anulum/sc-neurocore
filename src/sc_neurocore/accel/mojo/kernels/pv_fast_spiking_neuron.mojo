# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo SIMD RK4 kernel for the PV+ fast-spiking neuron

from math import exp

# State lanes hold (V, h, n, p) — exactly the four-wide SIMD vector.
alias State = SIMD[DType.float64, 4]


@always_inline
fn safe_rate(a: Float64, vhalf: Float64, v: Float64, k: Float64, fallback: Float64) -> Float64:
    """Wang-Buzsáki rate with the closed-form L'Hôpital limit near the singularity."""
    var d = v + vhalf
    if abs(d) < 1e-7:
        return fallback
    return a * d / (1.0 - exp(-d / k))


struct PVFastSpiking(Copyable, Movable):
    var g_na: Float64
    var g_k: Float64
    var g_kv3: Float64
    var g_l: Float64
    var e_na: Float64
    var e_k: Float64
    var e_l: Float64
    var c_m: Float64
    var phi: Float64
    var dt: Float64
    var v_threshold: Float64

    fn __init__(out self, g_kv3: Float64 = 5.0):
        self.g_na = 35.0
        self.g_k = 9.0
        self.g_kv3 = g_kv3
        self.g_l = 0.1
        self.e_na = 55.0
        self.e_k = -90.0
        self.e_l = -65.0
        self.c_m = 1.0
        self.phi = 5.0
        self.dt = 0.01
        self.v_threshold = -20.0

    fn derivatives(self, s: State, current: Float64) -> State:
        var v = s[0]
        var h = s[1]
        var n = s[2]
        var p = s[3]
        var am = safe_rate(0.1, 35.0, v, 10.0, 1.0)
        var bm = 4.0 * exp(-(v + 60.0) / 18.0)
        var m_inf = am / (am + bm)
        var ah = 0.07 * exp(-(v + 58.0) / 20.0)
        var bh = 1.0 / (1.0 + exp(-(v + 28.0) / 10.0))
        var an = safe_rate(0.01, 34.0, v, 10.0, 0.1)
        var bn = 0.125 * exp(-(v + 44.0) / 80.0)
        var p_inf = 1.0 / (1.0 + exp(-(v + 10.0) / 10.0))
        var i_na = self.g_na * m_inf * m_inf * m_inf * h * (v - self.e_na)
        var i_k = self.g_k * n * n * n * n * (v - self.e_k)
        var i_kv3 = self.g_kv3 * p * (v - self.e_k)
        var i_l = self.g_l * (v - self.e_l)
        var d = State(0.0)
        d[0] = (-i_na - i_k - i_kv3 - i_l + current) / self.c_m
        d[1] = self.phi * (ah * (1.0 - h) - bh * h)
        d[2] = self.phi * (an * (1.0 - n) - bn * n)
        d[3] = self.phi * (p_inf - p)
        return d

    fn rk4_substep(self, s: State, current: Float64) -> State:
        var dt = self.dt
        var k1 = self.derivatives(s, current)
        var k2 = self.derivatives(s + 0.5 * dt * k1, current)
        var k3 = self.derivatives(s + 0.5 * dt * k2, current)
        var k4 = self.derivatives(s + dt * k3, current)
        return s + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0

    fn simulate(self, n_steps: Int, current: Float64) -> Int:
        var n_sub = Int(0.5 / max(self.dt, 0.001))
        if n_sub < 1:
            n_sub = 1
        var s = State(0.0)
        s[0] = -65.0
        s[1] = 0.8
        s[2] = 0.1
        s[3] = 0.0
        var spikes = 0
        for _ in range(n_steps):
            var v_prev = s[0]
            for _ in range(n_sub):
                s = self.rk4_substep(s, current)
            if s[0] >= self.v_threshold and v_prev < self.v_threshold:
                spikes += 1
        return spikes


fn main():
    var neuron = PVFastSpiking()
    var spikes = neuron.simulate(40000, 2.0)
    print("pv_fast_spiking FS spikes:", spikes)

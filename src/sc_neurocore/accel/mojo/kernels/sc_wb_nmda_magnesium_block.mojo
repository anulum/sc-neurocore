# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo retained SC WB plus NMDA recurrence

from std.math import exp, isfinite


def _safe_rate(
    a: Float64, vhalf: Float64, v: Float64, k: Float64, fallback: Float64
) -> Float64:
    var delta = v + vhalf
    if abs(delta) < 1.0e-7:
        return fallback
    return a * delta / (1.0 - exp(-delta / k))


struct SCWBNMDAMagnesiumBlockNeuron(Copyable, Movable):
    var v: Float64
    var h: Float64
    var n: Float64
    var s_nmda: Float64
    var g_na: Float64
    var g_k: Float64
    var g_nmda: Float64
    var g_l: Float64
    var e_na: Float64
    var e_k: Float64
    var e_nmda: Float64
    var e_l: Float64
    var c_m: Float64
    var phi: Float64
    var mg_conc: Float64
    var tau_rise: Float64
    var tau_decay: Float64
    var dt: Float64
    var v_threshold: Float64
    var gain: Float64
    var sub_steps: Int

    def __init__(out self):
        self.v, self.h, self.n, self.s_nmda = -65.0, 0.6, 0.32, 0.0
        self.g_na, self.g_k, self.g_nmda, self.g_l = 35.0, 9.0, 0.5, 0.1
        self.e_na, self.e_k, self.e_nmda, self.e_l = 55.0, -90.0, 0.0, -65.0
        self.c_m, self.phi, self.mg_conc = 1.0, 5.0, 1.0
        self.tau_rise, self.tau_decay = 10.0, 100.0
        self.dt, self.v_threshold, self.gain = 0.5, -20.0, 1.0
        self.sub_steps = 50

    def valid(self) -> Bool:
        return (
            isfinite(self.v)
            and self.v >= -100.0
            and self.v <= 60.0
            and isfinite(self.h)
            and self.h >= 0.0
            and self.h <= 1.0
            and isfinite(self.n)
            and self.n >= 0.0
            and self.n <= 1.0
            and isfinite(self.s_nmda)
            and self.s_nmda >= 0.0
            and self.s_nmda <= 1.0
            and isfinite(self.g_na)
            and self.g_na >= 0.0
            and self.g_na <= 200.0
            and isfinite(self.g_k)
            and self.g_k >= 0.0
            and self.g_k <= 100.0
            and isfinite(self.g_nmda)
            and self.g_nmda >= 0.0
            and self.g_nmda <= 20.0
            and isfinite(self.g_l)
            and self.g_l >= 0.0
            and self.g_l <= 5.0
            and isfinite(self.e_na)
            and self.e_na >= 30.0
            and self.e_na <= 70.0
            and isfinite(self.e_k)
            and self.e_k >= -100.0
            and self.e_k <= -70.0
            and isfinite(self.e_nmda)
            and self.e_nmda >= -10.0
            and self.e_nmda <= 10.0
            and isfinite(self.e_l)
            and self.e_l >= -80.0
            and self.e_l <= -40.0
            and isfinite(self.c_m)
            and self.c_m >= 0.5
            and self.c_m <= 2.0
            and isfinite(self.phi)
            and self.phi >= 0.5
            and self.phi <= 10.0
            and isfinite(self.mg_conc)
            and self.mg_conc >= 0.0
            and self.mg_conc <= 5.0
            and isfinite(self.tau_rise)
            and self.tau_rise >= 0.1
            and self.tau_rise <= 20.0
            and isfinite(self.tau_decay)
            and self.tau_decay >= 10.0
            and self.tau_decay <= 500.0
            and isfinite(self.dt)
            and self.dt > 0.0
            and self.dt <= 1.0
            and isfinite(self.v_threshold)
            and self.v_threshold >= -20.0
            and self.v_threshold <= 20.0
            and isfinite(self.gain)
            and self.gain >= 0.0
            and self.gain <= 10.0
            and self.sub_steps >= 1
            and self.sub_steps <= 10000
        )

    def step(mut self, current: Float64) raises -> Int:
        if not isfinite(current) or not self.valid():
            raise Error("invalid retained SC NMDA state, configuration, or input")
        var input_current = self.gain * current
        var sub_dt = self.dt / Float64(self.sub_steps)
        var drive = 0.0
        if input_current > 0.0:
            drive = input_current / (input_current + 5.0)
        var tau = self.tau_decay
        if drive > self.s_nmda:
            tau = self.tau_rise
        var gate = max(
            0.0,
            min(1.0, self.s_nmda + self.dt * (drive - self.s_nmda) / tau),
        )
        var v, h, n = self.v, self.h, self.n
        var event = 0
        for _ in range(self.sub_steps):
            var alpha_m = _safe_rate(0.1, 35.0, v, 10.0, 1.0)
            var beta_m = 4.0 * exp(-(v + 60.0) / 18.0)
            var m_inf = alpha_m / (alpha_m + beta_m)
            var alpha_h = 0.07 * exp(-(v + 58.0) / 20.0)
            var beta_h = 1.0 / (1.0 + exp(-(v + 28.0) / 10.0))
            var alpha_n = _safe_rate(0.01, 34.0, v, 10.0, 0.1)
            var beta_n = 0.125 * exp(-(v + 44.0) / 80.0)
            var block = 1.0 / (1.0 + (self.mg_conc / 3.57) * exp(-0.062 * v))
            h += sub_dt * self.phi * (alpha_h * (1.0 - h) - beta_h * h)
            n += sub_dt * self.phi * (alpha_n * (1.0 - n) - beta_n * n)
            var i_na = self.g_na * m_inf * m_inf * m_inf * h * (v - self.e_na)
            var n_squared = n * n
            var i_k = self.g_k * n_squared * n_squared * (v - self.e_k)
            var i_nmda = self.g_nmda * gate * block * (v - self.e_nmda)
            var i_l = self.g_l * (v - self.e_l)
            v += sub_dt * (-i_na - i_k - i_nmda - i_l + input_current) / self.c_m
            if not isfinite(v) or not isfinite(h) or not isfinite(n):
                raise Error("retained SC NMDA candidate must be finite")
            if v >= self.v_threshold:
                event = 1
                v = -65.0
        self.v = max(-100.0, min(60.0, v))
        self.h = max(0.0, min(1.0, h))
        self.n = max(0.0, min(1.0, n))
        self.s_nmda = gate
        return event


def main() raises:
    var anchor = SCWBNMDAMagnesiumBlockNeuron()
    var event = anchor.step(5.0)
    print(event, anchor.v, anchor.h, anchor.n, anchor.s_nmda)

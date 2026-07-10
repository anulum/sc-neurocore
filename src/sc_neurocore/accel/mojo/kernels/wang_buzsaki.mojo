# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo kernel for wang_buzsaki
#
# Wang-Buzsáki (1996) fast-spiking hippocampal interneuron: a three-state simplified
# Hodgkin-Huxley (Na + delayed-rectifier K) with instantaneous sodium activation m = m_inf.
# Mirrors the Python golden sc_neurocore.neurons.models.wang_buzsaki.WangBuzsakiNeuron: a
# 0.5 ms macro step of 50 inner dt=0.01 sub-steps advanced sequentially (Gauss-Seidel — gates
# h/n from the old voltage, then voltage v from the new gates), rising-edge v >= v_threshold
# crossing on the macro boundary, no reset. DOI 10.1523/JNEUROSCI.16-20-06402.1996.
#
# Executable: `mojo run wang_buzsaki.mojo` runs the parity check in main() against the Python
# golden (three action potentials at I=10 over 20 macro steps; silent at zero current).

from std.math import exp


struct WangBuzsakiNeuron(Copyable, Movable):
    var v: Float64
    var h: Float64
    var n: Float64
    var g_na: Float64
    var g_k: Float64
    var g_l: Float64
    var e_na: Float64
    var e_k: Float64
    var e_l: Float64
    var c_m: Float64
    var phi: Float64
    var dt: Float64
    var v_threshold: Float64

    def __init__(out self):
        self.v = -65.0
        self.h = 0.8
        self.n = 0.1
        self.g_na = 35.0
        self.g_k = 9.0
        self.g_l = 0.1
        self.e_na = 55.0
        self.e_k = -90.0
        self.e_l = -65.0
        self.c_m = 1.0
        self.phi = 5.0
        self.dt = 0.01
        self.v_threshold = -20.0

    # Advance one 0.5 ms macro step (50 sequential sub-steps) and return 1 on a rising-edge
    # v >= v_threshold crossing, else 0. The gating variables are advanced from the old
    # voltage and the voltage from the new gates within each sub-step (Gauss-Seidel).
    def step(mut self, current: Float64) -> Int:
        var v_prev = self.v
        var v = self.v
        var h = self.h
        var n = self.n
        var substeps = Int(0.5 / self.dt)
        for _ in range(substeps):
            # alpha_m carries a removable singularity at v = -35 (limit 1.0); likewise
            # alpha_n at v = -34 (limit 0.1). m_inf = alpha_m / (alpha_m + beta_m).
            var dm = v + 35.0
            var abs_dm = dm if dm >= 0.0 else -dm
            var alpha_m: Float64
            if abs_dm > 1e-6:
                alpha_m = 0.1 * dm / (1.0 - exp(-dm / 10.0))
            else:
                alpha_m = 1.0
            var beta_m = 4.0 * exp(-(v + 60.0) / 18.0)
            var m_inf = alpha_m / (alpha_m + beta_m)
            var alpha_h = 0.07 * exp(-(v + 58.0) / 20.0)
            var beta_h = 1.0 / (1.0 + exp(-(v + 28.0) / 10.0))
            var dn = v + 34.0
            var abs_dn = dn if dn >= 0.0 else -dn
            var alpha_n: Float64
            if abs_dn > 1e-6:
                alpha_n = 0.01 * dn / (1.0 - exp(-dn / 10.0))
            else:
                alpha_n = 0.1
            var beta_n = 0.125 * exp(-(v + 44.0) / 80.0)
            var next_h = h + self.phi * (alpha_h * (1.0 - h) - beta_h * h) * self.dt
            var next_n = n + self.phi * (alpha_n * (1.0 - n) - beta_n * n) * self.dt
            var i_na = self.g_na * m_inf * m_inf * m_inf * next_h * (v - self.e_na)
            var i_k = self.g_k * next_n * next_n * next_n * next_n * (v - self.e_k)
            var i_l = self.g_l * (v - self.e_l)
            var next_v = v + (-i_na - i_k - i_l + current) / self.c_m * self.dt
            v = next_v
            h = next_h
            n = next_n
        self.v = v
        self.h = h
        self.n = n
        if self.v >= self.v_threshold and v_prev < self.v_threshold:
            return 1
        return 0


# Run the neuron for n_steps macro steps at constant current and return the spike count.
def simulate(n_steps: Int, current: Float64) -> Int:
    var neuron = WangBuzsakiNeuron()
    var spikes: Int = 0
    for _ in range(n_steps):
        spikes += neuron.step(current)
    return spikes


def main():
    # Parity contract against the Python golden: three action potentials at I=10 over 20 macro
    # steps (the count the gauss_seidel schema runner and the Q16.16 RTL reproduce three-way).
    var spikes = simulate(20, 10.0)
    print("I=10, 20 macro steps -> spikes =", spikes)
    if spikes == 3:
        print("PARITY OK (matches the Python golden: 3 action potentials)")
    else:
        print("PARITY FAIL: expected 3 to match the Python golden")
    var silent = simulate(20, 0.0)
    print("I=0 -> spikes =", silent, "(expect 0)")

# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo candidate-first RK4 kernel for dendritic NMDA

from std.math import exp


def dendritic_nmda_mg_block(v_dend: Float64, mg_conc: Float64) -> Float64:
    return 1.0 / (1.0 + (mg_conc / 3.57) * exp(-0.062 * v_dend))


def dendritic_nmda_rhs(v_soma: Float64, v_dend: Float64, i_soma: Float64,
                       glutamate: Float64) -> Tuple[Float64, Float64]:
    var g_nmda = 1.5
    var e_nmda = 0.0
    var mg_conc = 1.0
    var g_coupling = 0.5
    var tau_soma = 20.0
    var tau_dend = 50.0
    var block = dendritic_nmda_mg_block(v_dend, mg_conc)
    var i_nmda = g_nmda * glutamate * block * (v_dend - e_nmda)
    return ((-v_soma - 65.0 + i_soma + g_coupling * (v_dend - v_soma)) / tau_soma,
            (-v_dend - 65.0 - i_nmda + g_coupling * (v_soma - v_dend)) / tau_dend)


def dendritic_nmda_step(v_soma: Float64, v_dend: Float64, i_soma: Float64,
                        glutamate: Float64) -> Tuple[Float64, Float64, Int]:
    var dt = 0.1
    var theta = -50.0
    var k1s, k1d = dendritic_nmda_rhs(v_soma, v_dend, i_soma, glutamate)
    var k2s, k2d = dendritic_nmda_rhs(v_soma + 0.5 * dt * k1s, v_dend + 0.5 * dt * k1d,
                                      i_soma, glutamate)
    var k3s, k3d = dendritic_nmda_rhs(v_soma + 0.5 * dt * k2s, v_dend + 0.5 * dt * k2d,
                                      i_soma, glutamate)
    var k4s, k4d = dendritic_nmda_rhs(v_soma + dt * k3s, v_dend + dt * k3d,
                                      i_soma, glutamate)
    var next_soma = v_soma + dt * (k1s + 2.0 * k2s + 2.0 * k3s + k4s) / 6.0
    var next_dend = v_dend + dt * (k1d + 2.0 * k2d + 2.0 * k3d + k4d) / 6.0
    if next_soma >= theta:
        return (-65.0, next_dend, 1)
    return (next_soma, next_dend, 0)


def dendritic_nmda_simulate(n_steps: Int, i_soma: Float64,
                            glutamate: Float64) -> Tuple[Int, Float64, Float64]:
    var v_soma = -65.0
    var v_dend = -65.0
    var spikes = 0
    for _ in range(n_steps):
        var spike: Int
        v_soma, v_dend, spike = dendritic_nmda_step(v_soma, v_dend, i_soma, glutamate)
        spikes += spike
    return (spikes, v_soma, v_dend)


def main():
    var spikes, v_soma, v_dend = dendritic_nmda_simulate(20000, 50.0, 0.5)
    print("dendritic_nmda final state:", v_soma, v_dend)
    print("dendritic_nmda spikes:", spikes)

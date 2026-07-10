# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia Wang-Buzsáki parity test

# Parity contract for the Julia Wang-Buzsáki kernel: it must reproduce the Python golden
# (sc_neurocore.neurons.models.wang_buzsaki.WangBuzsakiNeuron) — three action potentials at
# I = 10 over 20 macro steps, the same count the gauss_seidel schema runner and the Q16.16 RTL
# reproduce three-way exactly — and stay silent at zero current. Lives beside the harness
# (not under neurons/) so test_neurons.jl's file-discovery loop does not try to load it as a
# kernel module. Run: julia src/sc_neurocore/accel/julia/wang_buzsaki_parity_test.jl

using Test

include(joinpath(@__DIR__, "neurons", "wang_buzsaki.jl"))
using .WangBuzsakiAccel

@testset "Wang-Buzsáki Julia kernel parity" begin
    # Faithful Gauss-Seidel dynamics: the sodium activation is instantaneous
    # (m_inf = alpha_m/(alpha_m+beta_m)), the gates h/n are advanced from the old voltage and
    # the voltage from the new gates, 50 sub-steps per 0.5 ms macro step, macro-boundary
    # v >= v_threshold crossing with no reset.
    _, spikes = WangBuzsakiAccel.simulate(20; I_ext=10.0)
    @test spikes == 3   # matches the Python golden three-way exactly

    _, silent = WangBuzsakiAccel.simulate(20; I_ext=0.0)
    @test silent == 0

    # Fail-closed: a non-finite input leaves the state untouched and reports the sentinel.
    s = WangBuzsakiAccel.WangBuzsakiNeuronState()
    before = (s.v, s.h, s.n)
    @test WangBuzsakiAccel.step!(s, NaN) == -1
    @test (s.v, s.h, s.n) == before
end

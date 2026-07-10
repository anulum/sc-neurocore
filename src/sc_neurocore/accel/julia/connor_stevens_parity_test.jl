# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia Connor-Stevens parity test

# Parity contract for the Julia Connor-Stevens kernel: it must reproduce the Python golden
# (sc_neurocore.neurons.models.connor_stevens.ConnorStevensNeuron, macro-step RK4 with 100
# sub-steps) — silent at zero drive, two action potentials at I = 10 over 100 macro steps, and
# nine at I = 20. The gating is exp-based, so the trace is not bit-exact across libms; the spike
# count is the stable observable and the Go and Rust kernels reproduce the same counts. Lives
# beside the harness (not under neurons/) so test_neurons.jl's file-discovery loop does not try
# to load it as a kernel module. Run: julia src/sc_neurocore/accel/julia/connor_stevens_parity_test.jl

using Test

include(joinpath(@__DIR__, "neurons", "connor_stevens.jl"))
using .ConnorStevensAccel

@testset "Connor-Stevens Julia kernel parity" begin
    for (current, want) in ((0.0, 0), (10.0, 2), (20.0, 9))
        _, spikes = ConnorStevensAccel.simulate(100; I_ext=current, dt=0.01)
        @test spikes == want
    end

    # Fail-closed: a non-finite input leaves the state untouched and reports the sentinel.
    s = ConnorStevensAccel.ConnorStevensNeuronState()
    before = (s.v, s.m, s.h, s.n, s.a, s.b)
    @test ConnorStevensAccel.step!(s, NaN) == -1
    @test (s.v, s.m, s.h, s.n, s.a, s.b) == before
end

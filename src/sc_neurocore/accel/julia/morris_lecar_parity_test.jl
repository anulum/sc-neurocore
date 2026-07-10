# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia Morris-Lecar parity test

# Parity contract for the Julia Morris-Lecar kernel: it must reproduce the Python golden
# (sc_neurocore.neurons.models.morris_lecar.MorrisLecarNeuron, single-step RK4) — silent at
# zero drive, three action potentials at I = 50 over 2000 steps, and five at I = 100. The
# gating is tanh/cosh, so the trace is not bit-exact across libms; the spike count is the
# stable observable and the Go and Rust kernels reproduce the same counts. Lives beside the
# harness (not under neurons/) so test_neurons.jl's file-discovery loop does not try to load
# it as a kernel module. Run: julia src/sc_neurocore/accel/julia/morris_lecar_parity_test.jl

using Test

include(joinpath(@__DIR__, "neurons", "morris_lecar.jl"))
using .MorrisLecarAccel

@testset "Morris-Lecar Julia kernel parity" begin
    for (current, want) in ((0.0, 0), (50.0, 3), (100.0, 5))
        _, spikes = MorrisLecarAccel.simulate(2000; I_ext=current, dt=0.1)
        @test spikes == want
    end

    # Fail-closed: a non-finite input leaves the state untouched and reports the sentinel.
    s = MorrisLecarAccel.MorrisLecarNeuronState()
    before = (s.v, s.w)
    @test MorrisLecarAccel.step!(s, NaN) == -1
    @test (s.v, s.w) == before
end

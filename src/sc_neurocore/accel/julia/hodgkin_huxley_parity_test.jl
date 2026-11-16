# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia Hodgkin-Huxley parity test

# Parity contract for the Julia Hodgkin-Huxley kernel: it must reproduce the Python golden
# (sc_neurocore.neurons.models.hodgkin_huxley.HodgkinHuxleyNeuron with the default baseline_euler
# integrator, 100 explicit-Euler sub-steps per macro step) — silent at zero drive, six action
# potentials at I = 10 over 100 macro steps, and nine at I = 20. The gating is exp-based, so the
# trace is not bit-exact across libms; the spike count is the stable observable and the Go and Rust
# kernels reproduce the same counts. Lives beside the harness (not under neurons/) so test_neurons.jl's
# file-discovery loop does not try to load it as a kernel module. Run:
#   julia src/sc_neurocore/accel/julia/hodgkin_huxley_parity_test.jl

using Test

include(joinpath(@__DIR__, "neurons", "hodgkin_huxley.jl"))
using .HodgkinHuxleyAccel

@testset "Hodgkin-Huxley Julia kernel parity" begin
    for (current, want) in ((0.0, 0), (10.0, 6), (20.0, 9))
        _, spikes = HodgkinHuxleyAccel.simulate(100; I_ext=current, dt=0.01)
        @test spikes == want
    end

    # Fail-closed: a non-finite input leaves the state untouched and reports the sentinel.
    s = HodgkinHuxleyAccel.HodgkinHuxleyNeuronState()
    before = (s.v, s.m, s.h, s.n)
    @test HodgkinHuxleyAccel.step!(s, NaN) == -1
    @test (s.v, s.m, s.h, s.n) == before

    # Fail-closed: a non-positive dt override is rejected without mutating state.
    s2 = HodgkinHuxleyAccel.HodgkinHuxleyNeuronState()
    before2 = (s2.v, s2.m, s2.h, s2.n)
    @test HodgkinHuxleyAccel.step!(s2, 10.0; dt=0.0) == -1
    @test (s2.v, s2.m, s2.h, s2.n) == before2

    # Where 1/dt is a half-integer, the Python model runs round(1/dt) sub-steps and Python rounds a
    # half to even: dt 0.4, 2/9 and 2/13 give 2, 4 and 6 sub-steps, and dt 2 none. Each state is the
    # Python model's after one macro step at I = 10 from the defaults.
    for (dt, python) in (
        (0.4, (-56.00630831519619, 0.09547477716734626, 0.5923767115876822, 0.3244914919575375)),
        (2 / 9, (-56.01081822375738, 0.10112513277783045, 0.5864793692830991, 0.32816421566612297)),
        (2 / 13, (-56.049500601211975, 0.10331990553856767, 0.5843154160461137, 0.32948115970033937)),
        (2.0, (-65.0, 0.05, 0.6, 0.32)),
    )
        s3 = HodgkinHuxleyAccel.HodgkinHuxleyNeuronState()
        @test HodgkinHuxleyAccel.step!(s3, 10.0; dt=dt) == 0
        @test all(abs.((s3.v, s3.m, s3.h, s3.n) .- python) .< 1e-9)
    end
end

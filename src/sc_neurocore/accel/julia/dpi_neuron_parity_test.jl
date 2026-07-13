# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia fidelity gate for the published DPI circuit

include(joinpath(@__DIR__, "neurons", "dpi_neuron.jl"))
using .DpiNeuronAccel

function factory_trace(steps::Int, current::Float64)
    return DpiNeuronAccel.simulate_trace(
        0.01,
        0.01,
        0.0,
        1.0,
        0.01,
        0.1,
        1.0,
        1.0,
        0.1,
        1.0,
        5.0,
        0.01,
        0.7,
        10.0,
        20.0,
        100.0,
        2.0,
        0.1,
        steps,
        current,
    )
end

one = factory_trace(1, 5.0)
@assert one.spikes == 0
@assert isapprox(one.i_mem_f, 0.010201975272610835; atol=1.0e-17, rtol=0.0)
@assert isapprox(one.i_ahp_f, 0.00999; atol=1.0e-17, rtol=0.0)
@assert one.refractory_time_f == 0.0

const GOLDENS = (
    (-0.1, 0),
    (0.0, 0),
    (1.0, 0),
    (2.0, 0),
    (3.0, 1),
    (5.0, 3),
    (10.0, 6),
    (20.0, 11),
    (50.0, 21),
)

for (current, expected) in GOLDENS
    result = factory_trace(1_000, current)
    @assert result.spikes == expected
    @assert length(result.trace) == 1_000
    @assert result.i_mem_f == result.trace[end]
    @assert isfinite(result.i_ahp_f)
    @assert result.refractory_time_f >= 0.0
end

configured = DpiNeuronAccel.simulate_trace(
    0.37,
    0.08,
    0.0,
    1.3,
    0.2,
    0.15,
    0.9,
    1.4,
    0.12,
    0.8,
    4.2,
    0.02,
    0.65,
    8.0,
    7.0,
    45.0,
    0.6,
    0.05,
    400,
    5.0,
)
@assert configured.spikes == 4
@assert length(configured.trace) == 400
@assert configured.i_mem_f == configured.trace[end] == 0.2
@assert isapprox(configured.i_ahp_f, 0.27412306389119817; atol=2.0e-15, rtol=0.0)
@assert configured.refractory_time_f == 0.0

empty = DpiNeuronAccel.simulate_trace(
    0.37,
    0.08,
    0.0,
    1.3,
    0.2,
    0.15,
    0.9,
    1.4,
    0.12,
    0.8,
    4.2,
    0.02,
    0.65,
    8.0,
    7.0,
    45.0,
    0.6,
    0.05,
    0,
    5.0,
)
@assert isempty(empty.trace)
@assert empty.spikes == 0
@assert (empty.i_mem_f, empty.i_ahp_f, empty.refractory_time_f) == (0.37, 0.08, 0.0)

pulse = DPINeuronState()
pulse.i_mem = 0.99
@assert step!(pulse, 10.0) == 1
@assert pulse.i_mem == pulse.i_reset
@assert pulse.refractory_time == pulse.refractory_period
before_ahp = pulse.i_ahp
@assert step!(pulse, 0.0) == 0
@assert pulse.i_mem == pulse.i_reset
@assert pulse.i_ahp > before_ahp

overflow = DPINeuronState()
overflow.tau = floatmin(Float64)
before = (overflow.i_mem, overflow.i_ahp, overflow.refractory_time)
try
    step!(overflow, floatmax(Float64))
    @assert false
catch error
    @assert error isa DomainError
end
@assert (overflow.i_mem, overflow.i_ahp, overflow.refractory_time) == before

reset_state = DPINeuronState(
    0.75,
    0.4,
    0.3,
    1.3,
    0.2,
    0.15,
    0.9,
    1.4,
    0.12,
    0.8,
    4.2,
    0.02,
    0.65,
    8.0,
    7.0,
    45.0,
    0.6,
    0.05,
)
reset!(reset_state)
@assert (reset_state.i_mem, reset_state.i_ahp, reset_state.refractory_time) == (0.2, 0.02, 0.0)
@assert reset_state.i_threshold == 1.3
@assert reset_state.tau == 7.0
@assert reset_state.dt == 0.05

println("DPI Julia parity assertions passed: ", length(GOLDENS) * 5 + 24)

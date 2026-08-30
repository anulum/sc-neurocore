# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Theta Julia executable parity gate

include(joinpath(@__DIR__, "neurons", "theta.jl"))
using .ThetaAccel

const GOLDENS = (
    (-1.0, 0),
    (-0.5, 0),
    (0.0, 0),
    (0.1, 1),
    (0.333, 2),
    (0.5, 2),
    (1.0, 3),
    (2.0, 5),
    (5.0, 7),
    (20.0, 14),
    (50.0, 23),
)

for (current, expected) in GOLDENS
    result = ThetaAccel.simulate_trace(0.0, 0.01, 1_000, current)
    @assert result.spikes == expected
    @assert length(result.trace) == 1_000
    @assert isfinite(result.thetaf)
end

configured = ThetaAccel.simulate_trace(0.37, 0.037, 400, 2.2)
@assert configured.spikes > 0
@assert length(configured.trace) == 400
@assert configured.thetaf == configured.trace[end]

complete = ThetaAccel.simulate_complete(0.37, 0.037, 400, 2.2)
@assert complete.trace == configured.trace
@assert sum(complete.events) == configured.spikes
@assert all(event -> event == 0 || event == 1, complete.events)
@assert complete.thetaf == configured.thetaf

empty = ThetaAccel.simulate_trace(0.37, 0.037, 0, 2.2)
@assert isempty(empty.trace)
@assert empty.spikes == 0
@assert empty.thetaf == ThetaAccel.wrap_phase(0.37)

rejected = ThetaNeuronState(0.25, 0.01)
try
    step!(rejected, -1.0e308; dt=1.0e308)
    @assert false
catch error
    @assert error isa DomainError
end
@assert rejected.theta == 0.25
@assert rejected.dt == 0.01

try
    ThetaAccel.simulate_complete(0.25, 1.0, 1, 16.0)
    @assert false
catch error
    @assert error isa DomainError
end

println("theta Julia parity assertions passed: ", length(GOLDENS) * 3 + 14)

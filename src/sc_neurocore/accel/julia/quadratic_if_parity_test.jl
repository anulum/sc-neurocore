# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Quadratic IF Julia executable parity gate

include(joinpath(@__DIR__, "neurons", "quadratic_if.jl"))
using .QuadraticIfAccel

const GOLDENS = (
    (0.0, 0),
    (0.333, 2),
    (0.5, 3),
    (1.0, 6),
    (2.0, 11),
    (5.0, 26),
    (20.0, 100),
    (50.0, 250),
)

for (current, expected) in GOLDENS
    result = QuadraticIfAccel.simulate_trace(-1.0, -1.0, 1.0, 0.01, 1_000, current)
    @assert result.spikes == expected
    @assert length(result.trace) == 1_000
    @assert isfinite(result.vf)
end

configured = QuadraticIfAccel.simulate_trace(-0.37, -1.3, 1.7, 0.037, 400, 2.2)
@assert configured.spikes > 0
@assert length(configured.trace) == 400
@assert configured.vf == configured.trace[end]

empty = QuadraticIfAccel.simulate_trace(-0.37, -1.3, 1.7, 0.037, 0, 2.2)
@assert isempty(empty.trace)
@assert empty.spikes == 0
@assert empty.vf == -0.37

rejected = QuadraticIFNeuronState()
try
    step!(rejected, -1.0e308; dt=1.0e308)
    @assert false
catch error
    @assert error isa DomainError
end
@assert rejected.v == -1.0
@assert rejected.dt == 0.01

println("quadratic-if Julia parity assertions passed: ", length(GOLDENS) * 3 + 9)

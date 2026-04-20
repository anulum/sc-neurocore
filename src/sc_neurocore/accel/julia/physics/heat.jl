# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for physics/heat

module HeatAccel

using Statistics, LinearAlgebra

mutable struct StochasticHeatSolverState
    length::Float64
    walkers::Float64
    alpha::Float64
end

function StochasticHeatSolverState()
    StochasticHeatSolverState(0.0, 0.0, 0.0)
end

function step(s::StochasticHeatSolverState)
    # Random step -1, 0, 1
    steps = np.random.choice([-1, 0, 1], size=length(s.walkers), p=[0.25, 0.5, 0.25])
    s.walkers += steps
    # Boundary conditions (Reflective)
    s.walkers = clamp(s.walkers, 0, s.length - 1)
end

function get_temperature_profile(s::StochasticHeatSolverState)
    density, _ = fit(Histogram, s.walkers, bins=s.length, range=(0, s.length))
    return density / length(s.walkers)
end

end # module HeatAccel

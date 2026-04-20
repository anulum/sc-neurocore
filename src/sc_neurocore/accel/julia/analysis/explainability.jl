# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for analysis/explainability

module ExplainabilityAccel

using Statistics, LinearAlgebra

mutable struct SpikeToConceptMapperState
    concept_map::Float64
end

function SpikeToConceptMapperState()
    SpikeToConceptMapperState(0.0)
end

function explain(s::SpikeToConceptMapperState, spikes, Any])
    active_indices = findall(spikes > 0)[0]
    concepts = []
    for idx in active_indices
        if idx in s.concept_map
            concepts = push!(, s.concept_map[idx])
        else
            concepts = push!(, f"Unknown({idx})")
    if ! concepts
        return "The agent is idle."
    return f"The agent is active on: {', '.join(concepts)}"
end

end # module ExplainabilityAccel

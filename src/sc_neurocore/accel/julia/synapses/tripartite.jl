# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for synapses/tripartite

module TripartiteAccel

using Statistics, LinearAlgebra

mutable struct TripartiteSynapseState
    base_weight::Float64
    glut_per_spike::Float64
    ca_threshold::Float64
    facilitation::Float64
    depression_rate::Float64
    w_min::Float64
    w_max::Float64
end

function TripartiteSynapseState()
    TripartiteSynapseState(0.5, 2.0, 0.3, 1.5, 0.001, 0.0, 1.0)
end

function step(s::TripartiteSynapseState, pre_spike, post_spike, dt)
    # Pre-synaptic activity → glutamate → IP3
    if pre_spike
        s._glut_current += s.glut_per_spike
    # Glutamate decays (tau_glut ~ 0.2s)
    s._glut_current *= math.exp(-dt / 0.2)
    # Step the astrocyte with glutamate-driven IP3 production
    s.astrocyte.dt = dt
    ca = s.astrocyte.step(s._glut_current)
    # Astrocyte modulation of synaptic weight
    if ca > s.ca_threshold
        # Gliotransmitter release → synaptic facilitation
        s.weight += s.facilitation * (ca - s.ca_threshold) * dt
    else
        # Slow depression toward baseline without astrocyte support
        s.weight += (s.base_weight - s.weight) * s.depression_rate
    s.weight = max(s.w_min, min(s.w_max, s.weight))
    return s.weight
end

function ca(s::TripartiteSynapseState)
    return s.astrocyte.ca
end

function ip3(s::TripartiteSynapseState)
    return s.astrocyte.ip3
end

function effective_weight(s::TripartiteSynapseState)
    return s.weight
end

function reset(s::TripartiteSynapseState)
    s.weight = s.base_weight
    s.astrocyte.reset()
    s._glut_current = 0.0
end

end # module TripartiteAccel

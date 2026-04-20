# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for robotics/swarm

module SwarmAccel

using Statistics, LinearAlgebra

mutable struct SwarmCouplingState
    coupling_strength::Float64
end

function SwarmCouplingState()
    SwarmCouplingState(0.1)
end

function synchronize(s::SwarmCouplingState, agent_a, agent_b)
    # We assume both agents have same number of neurons
    if agent_a.n_neurons != agent_b.n_neurons
        raise ValueError("Agents must have same size for direct coupling.")
    # Extract weights
    wa = agent_a.get_weights()
    wb = agent_b.get_weights()
    # Mutual Attraction: Weights shift toward each other
    # W_new = W + alpha * (W_other - W)
    delta = s.coupling_strength * (wb - wa)
    # Update Agent A
    new_wa = wa + delta
    for i in 1:agent_a.n_neurons
        for j in 1:agent_a.n_inputs
            agent_a.synapses[i][j].update_weight(new_wa[i, j])
    # Update Agent B (Reciprocal)
    new_wb = wb - delta
    for i in 1:agent_b.n_neurons
        for j in 1:agent_b.n_inputs
            agent_b.synapses[i][j].update_weight(new_wb[i, j])
    logger.info(
        "Swarm Synchronization: Shifted weights by magnitude %.6f", mean(abs(delta))
    )
end

end # module SwarmAccel

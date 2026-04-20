# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for synapses/gap_junction

module GapJunctionAccel

using Statistics, LinearAlgebra

mutable struct GapJunctionState
    conductance::Float64
    rectification::Float64
end

function GapJunctionState()
    GapJunctionState(0.1, 0.0)
end

function current(s::GapJunctionState, v_pre, v_post)
    dv = v_pre - v_post
    if s.rectification > 0
        # Rectification: reduce current in one direction
        factor = 1.0 - s.rectification * (1.0 if dv < 0 else 0.0)
        return s.conductance * dv * factor
    return s.conductance * dv
end

function current_matrix(s::GapJunctionState, voltages, adjacency)
    N = length(voltages)
    dv_matrix = voltages[np.newaxis, :] - voltages[:, np.newaxis]  # dv[i,j] = V[j] - V[i]
    currents = s.conductance * dv_matrix * adjacency
    return currents.sum(axis=1)
end

end # module GapJunctionAccel

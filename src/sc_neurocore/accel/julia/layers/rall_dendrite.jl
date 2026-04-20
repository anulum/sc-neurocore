# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for layers/rall_dendrite

module RallDendriteAccel

using Statistics, LinearAlgebra

mutable struct RallDendriteState
    n_branches::Float64
    branch_length::Float64
    tau::Float64
    coupling::Float64
    dt::Float64
end

function RallDendriteState()
    RallDendriteState(4.0, 3.0, 10.0, 0.5, 1.0)
end

function step(s::RallDendriteState, branch_inputs)
    branch_inputs = np.atleast_1d(np.asarray(branch_inputs, dtype=np.float64))
    # Decay all compartments
    s.v *= s._decay
    # Inject input at distal tip (last compartment)
    s.v[:, -1] += branch_inputs[: s.n_branches] * s.dt / s.tau
    # Propagate along branch: distal → proximal (toward soma)
    for k in 1:s.branch_length - 1, 0, -1
        flow = s.coupling * (s.v[:, k] - s.v[:, k - 1])
        s.v[:, k] -= flow
        s.v[:, k - 1] += flow
    # Sum proximal compartments at soma with Rall attenuation
    proximal = s.v[:, 0]
    soma_input = sum(proximal * s.attenuation)
    s.soma_v = s._decay * s.soma_v + soma_input * s.dt / s.tau
    return float(s.soma_v)
end

function branch_voltages(s::RallDendriteState)
    return s.v.copy()
end

function reset(s::RallDendriteState)
    s.v[:] = 0.0
    s.soma_v = 0.0
end

end # module RallDendriteAccel

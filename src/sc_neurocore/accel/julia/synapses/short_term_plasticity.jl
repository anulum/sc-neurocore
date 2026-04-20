# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for synapses/short_term_plasticity

module ShortTermPlasticityAccel

using Statistics, LinearAlgebra

mutable struct ShortTermPlasticitySynapseState
    x::Float64
    u::Float64
    u_base::Float64
    tau_d::Float64
    tau_f::Float64
    amplitude::Float64
    dt::Float64
end

function ShortTermPlasticitySynapseState()
    ShortTermPlasticitySynapseState(1.0, 0.5, 0.5, 200.0, 20.0, 1.0, 1.0)
end

function new_depressing(s::ShortTermPlasticitySynapseState)
    return cls(
        x=1.0,
        u=0.5,
        u_base=0.5,
        tau_d=200.0,
        tau_f=20.0,
        amplitude=1.0,
    )
end

function new_facilitating(s::ShortTermPlasticitySynapseState)
    return cls(
        x=1.0,
        u=0.1,
        u_base=0.1,
        tau_d=50.0,
        tau_f=500.0,
        amplitude=1.0,
    )
end

function step(s::ShortTermPlasticitySynapseState, pre_spike)
    # Recover between spikes.
    s.x += (1.0 - s.x) / s.tau_d * s.dt
    s.u += (s.u_base - s.u) / s.tau_f * s.dt
    if pre_spike
        # Facilitation: increase release probability.
        s.u += s.u_base * (1.0 - s.u)
        # Compute PSC before depression.
        psc = s.amplitude * s.u * s.x
        # Depression: consume resources.
        s.x -= s.u * s.x
        s.x = max(s.x, 0.0)
        return psc
    return 0.0
end

function reset(s::ShortTermPlasticitySynapseState)
    s.x = 1.0
    s.u = s.u_base
end

end # module ShortTermPlasticityAccel

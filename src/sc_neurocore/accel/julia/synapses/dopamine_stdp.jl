# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for synapses/dopamine_stdp

module DopamineStdpAccel

using Statistics, LinearAlgebra

mutable struct DopamineStdpSynapseState
    weight::Float64
    w_min::Float64
    w_max::Float64
    tau_e::Float64
    tau_da::Float64
    tau_pre::Float64
    tau_post::Float64
    a_plus::Float64
    a_minus::Float64
    lr::Float64
    dt::Float64
    eligibility::Float64
    dopamine::Float64
    trace_pre::Float64
    trace_post::Float64
end

function DopamineStdpSynapseState()
    DopamineStdpSynapseState(0.5, 0.0, 1.0, 1000.0, 200.0, 20.0, 20.0, 1.0, -1.0, 0.001, 1.0, 0.0, 0.0, 0.0, 0.0)
end

function step(s::DopamineStdpSynapseState, pre_spike, post_spike, reward)
    # Decay traces.
    s.trace_pre *= math.exp(-s.dt / s.tau_pre)
    s.trace_post *= math.exp(-s.dt / s.tau_post)
    s.eligibility *= math.exp(-s.dt / s.tau_e)
    s.dopamine += (-s.dopamine / s.tau_da + reward) * s.dt
    if pre_spike
        # LTD from accumulated post-trace.
        s.eligibility += s.a_minus * s.trace_post
        s.trace_pre += 1.0
    if post_spike
        # LTP from accumulated pre-trace.
        s.eligibility += s.a_plus * s.trace_pre
        s.trace_post += 1.0
    # Dopamine-gated weight update.
    dw = s.lr * s.dopamine * s.eligibility * s.dt
    s.weight = max(s.w_min, min(s.w_max, s.weight + dw))
    return s.weight
end

function reset(s::DopamineStdpSynapseState)
    s.eligibility = 0.0
    s.dopamine = 0.0
    s.trace_pre = 0.0
    s.trace_post = 0.0
end

end # module DopamineStdpAccel

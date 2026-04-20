# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for synapses/r_stdp

module RStdpAccel

using Statistics, LinearAlgebra

mutable struct RewardModulatedSTDPSynapseState
    eligibility_trace::Float64
    trace_decay::Float64
    anti_hebbian_scale::Float64
end

function RewardModulatedSTDPSynapseState()
    RewardModulatedSTDPSynapseState(0.0, 0.0, 0.0)
end

function process_step(s::RewardModulatedSTDPSynapseState, pre_bit, post_bit)
    # 1. Compute Output (Same as standard)
    w_prob = s.effective_weight_probability()
    weight_bit = 1 if s._rng.random() < w_prob else 0
    output_bit = pre_bit & weight_bit
    # 2. Update Eligibility Trace instead of Weight
    # (Simplified Hebbian / STDP logic)
    # Hebbian Term: Pre * Post
    # If both fire, trace goes up (Potentiation eligibility)
    if pre_bit == 1 && post_bit == 1
        s.eligibility_trace += 1.0
    # Anti-Hebbian Term: Pre * !Post (|| vice versa depending on rule)
    # If Pre fires but Post doesn't, trace goes down (Depression eligibility)
    elseif pre_bit == 1 && post_bit == 0
        s.eligibility_trace -= s.anti_hebbian_scale
    # Decay trace
    s.eligibility_trace *= s.trace_decay
    return output_bit
end

function apply_reward(s::RewardModulatedSTDPSynapseState, reward)
    # Delta W ~ Reward * Trace
    update = s.learning_rate * reward * s.eligibility_trace
    new_w = s.w + update
    # Clip
    new_w = max(s.w_min, min(s.w_max, new_w))
    s.update_weight(new_w)
end

end # module RStdpAccel

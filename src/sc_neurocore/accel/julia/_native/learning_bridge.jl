# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for _native/learning_bridge

module LearningBridgeAccel

const _LIB_PATH = joinpath(dirname(dirname(dirname(@__DIR__))), "_native", "libautonomous_learning.so")
const _HAS_LEARNING = isfile(_LIB_PATH)

# Rule Type Constants
const RULE_ELIGENT = 0
const RULE_STDP = 1
const RULE_REWARD_STDP = 2
const RULE_BCM = 3

mutable struct RustPlasticityRule
    rule_type::UInt32
    ptr::Ptr{Cvoid}

    function RustPlasticityRule(rule_type::Int=RULE_STDP, weight::Float32=0.5f0, param_a::Float32=0.01f0, param_b::Float32=0.012f0)
        if !_HAS_LEARNING
            error("libautonomous_learning.so not available")
        end
        ptr = ccall((:create_rule, _LIB_PATH), Ptr{Cvoid}, (UInt32, Float32, Float32, Float32), UInt32(rule_type), weight, param_a, param_b)
        obj = new(UInt32(rule_type), ptr)
        finalizer(destroy_rule, obj)
        return obj
    end
end

function destroy_rule(rule::RustPlasticityRule)
    if rule.ptr != C_NULL && _HAS_LEARNING
        ccall((:destroy_rule, _LIB_PATH), Cvoid, (Ptr{Cvoid},), rule.ptr)
        rule.ptr = C_NULL
    end
end

function is_available()
    return _HAS_LEARNING
end

function step(s::RustPlasticityRule, pre_spike::Bool, post_spike::Bool, reward::Float32=0.0f0)
    ccall((:step_rule, _LIB_PATH), Cvoid, (Ptr{Cvoid}, Bool, Bool, Float32), s.ptr, pre_spike, post_spike, reward)
end

function step_batched(s::RustPlasticityRule, pre_spikes::Array{Bool, 1}, post_spikes::Array{Bool, 1}, rewards::Array{Float32, 1})
    count = length(pre_spikes)
    ccall((:step_rule_batched, _LIB_PATH), Cvoid, (Ptr{Cvoid}, Ptr{Bool}, Ptr{Bool}, Ptr{Float32}, UInt), s.ptr, pre_spikes, post_spikes, rewards, count)
end

function weight(s::RustPlasticityRule)
    return ccall((:get_rule_weight, _LIB_PATH), Float32, (Ptr{Cvoid},), s.ptr)
end

function reset(s::RustPlasticityRule)
    ccall((:reset_rule, _LIB_PATH), Cvoid, (Ptr{Cvoid},), s.ptr)
end

mutable struct RustEligentLearner
    ptr::Ptr{Cvoid}

    function RustEligentLearner(threshold::Float32=1.0f0, target_rate::Float32=0.1f0, weight::Float32=0.5f0)
        if !_HAS_LEARNING
            error("libautonomous_learning.so not available")
        end
        ptr = ccall((:create_learner, _LIB_PATH), Ptr{Cvoid}, (Float32, Float32, Float32), threshold, target_rate, weight)
        obj = new(ptr)
        finalizer(destroy_learner, obj)
        return obj
    end
end

function destroy_learner(learner::RustEligentLearner)
    if learner.ptr != C_NULL && _HAS_LEARNING
        ccall((:destroy_learner, _LIB_PATH), Cvoid, (Ptr{Cvoid},), learner.ptr)
        learner.ptr = C_NULL
    end
end

function step(s::RustEligentLearner, fired::Bool, pre_spike::Bool, global_reward::Float32=0.0f0)
    ccall((:step_learner, _LIB_PATH), Cvoid, (Ptr{Cvoid}, Bool, Bool, Float32), s.ptr, fired, pre_spike, global_reward)
end

function step_batched(s::RustEligentLearner, fired::Array{Bool, 1}, pre_spikes::Array{Bool, 1}, global_rewards::Array{Float32, 1})
    count = length(fired)
    ccall((:step_learner_batched, _LIB_PATH), Cvoid, (Ptr{Cvoid}, Ptr{Bool}, Ptr{Bool}, Ptr{Float32}, UInt), s.ptr, fired, pre_spikes, global_rewards, count)
end

end # module LearningBridgeAccel

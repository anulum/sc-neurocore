# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for _native/learning_bridge

module LearningBridgeAccel

const _DEFAULT_LIB_PATH = joinpath(dirname(dirname(dirname(@__DIR__))), "_native", "libautonomous_learning.so")
const _LIB_PATH = get(ENV, "SC_NEUROCORE_LIB_PATH", _DEFAULT_LIB_PATH)
const _HAS_LEARNING = isfile(_LIB_PATH)

# Rule Type Constants
const RULE_ELIGENT = 0
const RULE_STDP = 1
const RULE_REWARD_STDP = 2
const RULE_BCM = 3

function require_rule_type(rule_type::Int)
    rule_type in RULE_ELIGENT:RULE_BCM || throw(ArgumentError("rule_type must be in 0:3"))
    return UInt32(rule_type)
end

function require_finite(name::String, value::Float32)
    isfinite(value) || throw(ArgumentError("$name must be finite"))
    return value
end

function require_nonnegative(name::String, value::Float32)
    require_finite(name, value)
    value >= 0.0f0 || throw(ArgumentError("$name must be non-negative"))
    return value
end

function require_positive(name::String, value::Float32)
    require_finite(name, value)
    value > 0.0f0 || throw(ArgumentError("$name must be positive"))
    return value
end

function require_weight(weight::Float32)
    require_finite("weight", weight)
    0.0f0 <= weight <= 1.0f0 || throw(ArgumentError("weight must be in [0, 1]"))
    return weight
end

function require_live(name::String, ptr::Ptr{Cvoid})
    ptr != C_NULL || throw(ArgumentError("$name is closed"))
end

function require_equal_nonempty(vectors...)
    lengths = length.(vectors)
    isempty(lengths) && throw(ArgumentError("at least one vector is required"))
    lengths[1] > 0 || throw(ArgumentError("learning vectors must not be empty"))
    all(length -> length == lengths[1], lengths) ||
        throw(DimensionMismatch("learning vectors must have equal lengths"))
    return lengths[1]
end

mutable struct RustPlasticityRule
    rule_type::UInt32
    ptr::Ptr{Cvoid}

    function RustPlasticityRule(rule_type::Int=RULE_STDP, weight::Float32=0.5f0, param_a::Float32=0.01f0, param_b::Float32=0.012f0)
        if !_HAS_LEARNING
            error("libautonomous_learning.so not available")
        end
        validated_rule = require_rule_type(rule_type)
        ptr = ccall(
            (:create_rule, _LIB_PATH),
            Ptr{Cvoid},
            (UInt32, Float32, Float32, Float32),
            validated_rule,
            require_weight(weight),
            require_nonnegative("param_a", param_a),
            require_nonnegative("param_b", param_b),
        )
        ptr != C_NULL || error("Rust plasticity-rule construction failed")
        obj = new(validated_rule, ptr)
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

function step(s::RustPlasticityRule, pre_spike::Bool, post_spike::Bool, reward::Float32=0.0f0, dt::Float32=0.001f0)
    require_live("plasticity rule", s.ptr)
    ccall(
        (:step_rule, _LIB_PATH),
        Cvoid,
        (Ptr{Cvoid}, Bool, Bool, Float32, Float32),
        s.ptr,
        pre_spike,
        post_spike,
        require_finite("reward", reward),
        require_positive("dt", dt),
    )
end

function step_batched(s::RustPlasticityRule, pre_spikes::Array{Bool, 1}, post_spikes::Array{Bool, 1}, rewards::Array{Float32, 1}, dt::Float32=0.001f0)
    require_live("plasticity rule", s.ptr)
    count = require_equal_nonempty(pre_spikes, post_spikes, rewards)
    all(isfinite, rewards) || throw(ArgumentError("rewards must be finite"))
    ccall(
        (:step_rule_batched, _LIB_PATH),
        Cvoid,
        (Ptr{Cvoid}, Ptr{Bool}, Ptr{Bool}, Ptr{Float32}, UInt, Float32),
        s.ptr,
        pre_spikes,
        post_spikes,
        rewards,
        count,
        require_positive("dt", dt),
    )
end

function weight(s::RustPlasticityRule)
    require_live("plasticity rule", s.ptr)
    return ccall((:get_rule_weight, _LIB_PATH), Float32, (Ptr{Cvoid},), s.ptr)
end

function reset(s::RustPlasticityRule)
    require_live("plasticity rule", s.ptr)
    ccall((:reset_rule, _LIB_PATH), Cvoid, (Ptr{Cvoid},), s.ptr)
end

mutable struct RustEligentLearner
    ptr::Ptr{Cvoid}

    function RustEligentLearner(threshold::Float32=1.0f0, target_rate::Float32=0.1f0, weight::Float32=0.5f0)
        if !_HAS_LEARNING
            error("libautonomous_learning.so not available")
        end
        ptr = ccall(
            (:create_learner, _LIB_PATH),
            Ptr{Cvoid},
            (Float32, Float32, Float32),
            require_positive("threshold", threshold),
            require_nonnegative("target_rate", target_rate),
            require_weight(weight),
        )
        ptr != C_NULL || error("Rust ELIGENT learner construction failed")
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

function step(s::RustEligentLearner, fired::Bool, pre_spike::Bool, global_reward::Float32=0.0f0, dt::Float32=0.001f0)
    require_live("ELIGENT learner", s.ptr)
    ccall(
        (:step_learner, _LIB_PATH),
        Cvoid,
        (Ptr{Cvoid}, Bool, Bool, Float32, Float32),
        s.ptr,
        fired,
        pre_spike,
        require_finite("global_reward", global_reward),
        require_positive("dt", dt),
    )
end

function step_batched(s::RustEligentLearner, fired::Array{Bool, 1}, pre_spikes::Array{Bool, 1}, global_rewards::Array{Float32, 1}, dt::Float32=0.001f0)
    require_live("ELIGENT learner", s.ptr)
    count = require_equal_nonempty(fired, pre_spikes, global_rewards)
    all(isfinite, global_rewards) || throw(ArgumentError("global_rewards must be finite"))
    ccall(
        (:step_learner_batched, _LIB_PATH),
        Cvoid,
        (Ptr{Cvoid}, Ptr{Bool}, Ptr{Bool}, Ptr{Float32}, UInt, Float32),
        s.ptr,
        fired,
        pre_spikes,
        global_rewards,
        count,
        require_positive("dt", dt),
    )
end

end # module LearningBridgeAccel

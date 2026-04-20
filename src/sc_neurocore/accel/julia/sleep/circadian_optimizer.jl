# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for sleep/circadian_optimizer

module CircadianOptimizerAccel

using Statistics, LinearAlgebra

mutable struct CircadianOptimizerState
    chronotype::Float64
    bedtime_hour::Float64
    wake_hour::Float64
    default_protocol::Float64
    melatonin_peak_hour::Float64
    core_body_temp_nadir_hour::Float64
    _profile::Float64
end

function CircadianOptimizerState()
    CircadianOptimizerState(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
end

function get_profile(s::CircadianOptimizerState)
    return s._profile
end

function get_sleep_window(s::CircadianOptimizerState)
    return (s._profile.bedtime_hour, s._profile.wake_hour)
end

function get_recommended_protocol(s::CircadianOptimizerState)
    return s._profile.default_protocol
end

function is_in_sleep_window(s::CircadianOptimizerState, hour)
    bed = s._profile.bedtime_hour
    wake = s._profile.wake_hour
    if bed <= wake
        return bed <= hour < wake
    else
        # wraps past midnight
        return hour >= bed || hour < wake
end

function melatonin_level(s::CircadianOptimizerState, hour)
    peak = s._profile.melatonin_peak_hour
    # phase so that cos(0) = 1 at the peak hour
    phase = 2.0 * math.pi * (hour - peak) / 24.0
    level = 0.5 * (1.0 + math.cos(phase))
    return float(clamp(level, 0.0, 1.0))
end

function to_dict(s::CircadianOptimizerState)
    p = s._profile
    return {
        "chronotype": s.chronotype.value,
        "bedtime_hour": p.bedtime_hour,
        "wake_hour": p.wake_hour,
        "default_protocol": p.default_protocol,
        "melatonin_peak_hour": p.melatonin_peak_hour,
        "core_body_temp_nadir_hour": p.core_body_temp_nadir_hour,
    }
end

end # module CircadianOptimizerAccel

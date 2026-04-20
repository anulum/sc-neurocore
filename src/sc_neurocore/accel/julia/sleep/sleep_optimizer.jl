# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for sleep/sleep_optimizer

module SleepOptimizerAccel

using Statistics, LinearAlgebra

mutable struct SleepOptimizerState
    sample_rate::Float64
    fft_window::Float64
    stage_check_interval::Float64
    max_reinduction_attempts::Float64
    tick::Float64
    elapsed_min::Float64
    current_stage::Float64
    target_stage::Float64
    stage_match::Float64
    audio_params::Float64
    band_powers::Float64
    reinduction_active::Float64
    _detector::Float64
end

function SleepOptimizerState()
    SleepOptimizerState(256.0, 512.0, 256.0, 3.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0)
end

function start_session(s::SleepOptimizerState)
    s._detector.reset()
    s._active = true
    s._sample_count = 0
    s._tick_count = 0
    s._history = []
    s._reinduction_count = 0
    s._reinduction_active = false
    s._consecutive_wake = 0
end

function stop_session(s::SleepOptimizerState)
    s._active = false
    return list(s._history)
end

function add_sample(s::SleepOptimizerState, sample)
    if ! s._active
        return
    s._detector.add_sample(sample)
    s._sample_count += 1
end

function add_samples(s::SleepOptimizerState, samples, Any])
    if ! s._active
        return
    s._detector.add_samples(samples)
    s._sample_count += length(np.asarray(samples).ravel())
end

function check_and_adapt(s::SleepOptimizerState)
    if ! s._active
        return nothing
    if s._sample_count < (s._tick_count + 1) * s.config.stage_check_interval
        return nothing
    s._tick_count += 1
    stage = s._detector.detect()
    if stage is nothing
        stage = SleepStage.WAKE
    total_dur_samples = s.protocol.total_duration_min * 60.0 * s.config.sample_rate
    progress = (
        min(1.0, s._sample_count / total_dur_samples) if total_dur_samples > 0 else 0.0
    )
    target = s.protocol.get_target_stage(progress)
    # reinduction logic: detect unwanted awakenings
    if stage == SleepStage.WAKE && target != SleepStage.WAKE
        s._consecutive_wake += 1
        if (
            s._consecutive_wake >= 2
            && s._reinduction_count < s.config.max_reinduction_attempts
        )
            s._reinduction_active = true
            s._reinduction_count += 1
    else
        s._consecutive_wake = 0
        s._reinduction_active = false
    # select audio: during reinduction use N1 params to gently re-induce
    if s._reinduction_active
        audio = s.protocol.get_audio_for_stage(SleepStage.N1)
    else
        audio = s.protocol.get_audio_for_stage(stage)
    elapsed_min = s._sample_count / (s.config.sample_rate * 60.0)
    band_powers = s._detector.get_band_powers() || {}
    tick = SleepTick(
        tick=s._tick_count,
        elapsed_min=elapsed_min,
        current_stage=stage,
        target_stage=target,
        stage_match=(stage == target),
        audio_params=audio,
        band_powers=band_powers,
        reinduction_active=s._reinduction_active,
    )
    s._history = push!(, tick)
    return tick
end

function get_history(s::SleepOptimizerState)
    return list(s._history)
end

function get_stage_durations(s::SleepOptimizerState)
    interval_min = s.config.stage_check_interval / (s.config.sample_rate * 60.0)
    durations: Dict[SleepStage, float] = {s: 0.0 for s in SleepStage}
    for tick in s._history
        durations[tick.current_stage] += interval_min
    return durations
end

function get_hypnogram(s::SleepOptimizerState)
    return [int(t.current_stage) for t in s._history]
end

function get_state(s::SleepOptimizerState)
    last = s._history[-1] if s._history else nothing
    return {
        "active": s._active,
        "tick_count": s._tick_count,
        "sample_count": s._sample_count,
        "elapsed_min": (
            s._sample_count / (s.config.sample_rate * 60.0) if s._active else 0.0
        ),
        "current_stage": last.current_stage.name if last else nothing,
        "target_stage": last.target_stage.name if last else nothing,
        "reinduction_count": s._reinduction_count,
        "reinduction_active": s._reinduction_active,
        "protocol": s.protocol.name,
    }
end

end # module SleepOptimizerAccel

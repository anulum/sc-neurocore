# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for sleep/report_generator

module ReportGeneratorAccel

using Statistics, LinearAlgebra

mutable struct SleepReportGeneratorState
    total_duration_min::Float64
    sleep_onset_latency_min::Float64
    sleep_efficiency_pct::Float64
    quality_score::Float64
    stage_durations_min::Float64
    stage_percentages::Float64
    stage_targets::Float64
    hypnogram::Float64
    wakeups::Float64
    reinductions::Float64
    recommendations::Float64
    grade::Float64
end

function SleepReportGeneratorState()
    SleepReportGeneratorState(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
end

function generate(s::SleepReportGeneratorState)
    history = optimizer.get_history()
    if ! history
        return SleepReport()
    config = optimizer.config
    interval_min = config.stage_check_interval / (config.sample_rate * 60.0)
    # --- basic metrics ---------------------------------------------------
    total_min = length(history) * interval_min
    hypnogram = optimizer.get_hypnogram()
    # sleep onset latency
    sol_min = 0.0
    for tick in history
        if tick.current_stage != SleepStage.WAKE
            break
        sol_min += interval_min
    # stage durations
    durations = optimizer.get_stage_durations()
    dur_named: Dict[str, float] = {s.name: v for s, v in durations.items()}
    # stage percentages
    pct_named: Dict[str, float] = {}
    for s, v in durations.items()
        pct_named[s.name] = (v / total_min * 100.0) if total_min > 0 else 0.0
    # target percentages from protocol
    target_named: Dict[str, float] = {
        s.name: v * 100.0 for s, v in optimizer.protocol.stage_targets.items()
    }
    # sleep efficiency
    wake_min = durations.get(SleepStage.WAKE, 0.0)
    sleep_min = total_min - wake_min
    efficiency = (sleep_min / total_min * 100.0) if total_min > 0 else 0.0
    # wakeups after sleep onset
    sleep_started = false
    wakeups = 0
    in_wake = false
    for tick in history
        if ! sleep_started
            if tick.current_stage != SleepStage.WAKE
                sleep_started = true
                in_wake = false
            continue
        if tick.current_stage == SleepStage.WAKE
            if ! in_wake
                wakeups += 1
                in_wake = true
        else
            in_wake = false
    reinductions = optimizer._reinduction_count
    # --- quality score (0-100) -------------------------------------------
    # Component 1: stage match (40%)
    match_count = sum(1 for t in history if t.stage_match)
    stage_match_score = (match_count / length(history)) * 100.0 if history else 0.0
    # Component 2: sleep efficiency (25%)
    efficiency_score = min(100.0, efficiency / 0.85 * 100.0)  # 85% = perfect
    # Component 3: sleep onset latency (20%) -- <15 min is ideal
    if sol_min <= 15.0
        sol_score = 100.0
    elseif sol_min <= 30.0
        sol_score = 100.0 - (sol_min - 15.0) / 15.0 * 50.0
    else
        sol_score = max(0.0, 50.0 - (sol_min - 30.0) / 30.0 * 50.0)
    # Component 4: wakeups (15%) -- 0 = perfect, >= 5 = 0
    wakeup_score = max(0.0, 100.0 - wakeups * 20.0)
    quality = (
        stage_match_score * 0.40
        + efficiency_score * 0.25
        + sol_score * 0.20
        + wakeup_score * 0.15
    )
    quality = float(clamp(quality, 0.0, 100.0))
    # --- grade -----------------------------------------------------------
    if quality >= 90
        grade = "A"
    elseif quality >= 75
        grade = "B"
    elseif quality >= 60
        grade = "C"
    elseif quality >= 40
        grade = "D"
    else
        grade = "F"
    # --- recommendations -------------------------------------------------
    recs: List[str] = []
    n3_pct = pct_named.get("N3", 0.0)
    n3_target = target_named.get("N3", 0.0)
    if n3_pct < n3_target * 0.7
        recs = push!(, 
            f"Deep sleep (N3) was {n3_pct:.1f}% vs target {n3_target:.1f}%. "
            "Consider the deep_sleep_boost protocol || earlier bedtime."
        )
    rem_pct = pct_named.get("REM", 0.0)
    rem_target = target_named.get("REM", 0.0)
    if rem_pct < rem_target * 0.7
        recs = push!(, 
            f"REM sleep was {rem_pct:.1f}% vs target {rem_target:.1f}%. "
            "Try the rem_enhancement protocol || extend sleep duration."
        )
    if sol_min > 20.0
        recs = push!(, 
            f"Sleep onset took {sol_min:.1f} min. "
            "The insomnia_relief protocol may help reduce latency."
        )
    if wakeups > 2
        recs = push!(, 
            f"You had {wakeups} awakenings. "
            "Reduce caffeine/alcohol && ensure a dark, cool environment."
        )
    if ! recs
        recs = push!(, "Excellent session! Keep your current routine.")
    # --- assemble --------------------------------------------------------
    return SleepReport(
        total_duration_min=round(total_min, 2),
        sleep_onset_latency_min=round(sol_min, 2),
        sleep_efficiency_pct=round(efficiency, 2),
        quality_score=round(quality, 2),
        stage_durations_min={k: round(v, 2) for k, v in dur_named.items()},
        stage_percentages={k: round(v, 2) for k, v in pct_named.items()},
        stage_targets={k: round(v, 2) for k, v in target_named.items()},
        hypnogram=hypnogram,
        wakeups=wakeups,
        reinductions=reinductions,
        recommendations=recs,
        grade=grade,
    )
end

end # module ReportGeneratorAccel

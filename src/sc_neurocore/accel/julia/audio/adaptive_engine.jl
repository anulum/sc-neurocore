# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for audio/adaptive_engine

module AdaptiveEngineAccel

using Statistics, LinearAlgebra

mutable struct AdaptiveAudioEngineState
    tick::Float64
    phase::Float64
    param::Float64
    old_value::Float64
    new_value::Float64
    reason::Float64
    total_ticks::Float64
    avg_evs::Float64
    peak_evs::Float64
    verified_pct::Float64
    grade::Float64
    adaptations::Float64
    phase_durations::Float64
    final_audio::Float64
    ssgf::Float64
end

function AdaptiveAudioEngineState()
    AdaptiveAudioEngineState(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
end

function to_dict(s::AdaptiveAudioEngineState)
    return {
        "total_ticks": s.total_ticks,
        "avg_evs": round(s.avg_evs, 2),
        "peak_evs": round(s.peak_evs, 2),
        "verified_pct": round(s.verified_pct, 2),
        "grade": s.grade,
        "adaptations": s.adaptations,
        "phase_durations": s.phase_durations,
        "final_audio": s.final_audio,
    }
end

function _update_phase(s::AdaptiveAudioEngineState)
    if s._phase == SessionPhase.DISCOVERY && s._tick >= _DISCOVERY_TICKS
        s._phase = SessionPhase.LOCK_ON
        s._phase_start_tick = s._tick
        logger.info("Session phase -> LOCK_ON at tick %d", s._tick)
    elseif s._phase == SessionPhase.LOCK_ON && s._tick >= _LOCKON_TICKS
        s._phase = SessionPhase.DEEPENING
        s._phase_start_tick = s._tick
        logger.info("Session phase -> DEEPENING at tick %d", s._tick)
end

function _evs_trend(s::AdaptiveAudioEngineState)
    if length(s._recent_evs) < 3
        return 0.0
    recent = collect(s._recent_evs[-s._trend_window :])
    if length(recent) < 3
        return 0.0
    # Simple linear slope
    x = collect(length(recent), dtype=np.float64)
    x_mean = x.mean()
    y_mean = recent.mean()
    denom = sum((x - x_mean) ^ 2)
    if denom < 1e-12
        return 0.0
    slope = sum((x - x_mean) * (recent - y_mean)) / denom
    return float(slope)
end

function on_evs_update(s::AdaptiveAudioEngineState, snapshot)
    s._tick += 1
    s._update_phase()
    # Track EVS
    score = snapshot.evs_score
    s._evs_scores = push!(, score)
    s._recent_evs = push!(, score)
    if length(s._recent_evs) > s._trend_window * 2
        s._recent_evs = s._recent_evs[-s._trend_window * 2 :]
    if snapshot.is_verified
        s._verified_count += 1
    trend = s._evs_trend()
    # Phase-specific adaptation
    if s._phase == SessionPhase.DISCOVERY
        s._adapt_discovery(snapshot, trend)
    elseif s._phase == SessionPhase.LOCK_ON
        s._adapt_lock_on(snapshot, trend)
    else
        s._adapt_deepening(snapshot, trend)
    # Run one SSGF outer step to update geometry
    s.ssgf.outer_step()
    return s.ssgf.get_audio_mapping()
end

function _adapt_discovery(s::AdaptiveAudioEngineState, snap, trend)
    cfg = s.ssgf.cfg
    # Sweep target Hz slowly
    s._sweep_hz += s._sweep_direction * 0.1
    if s._sweep_hz > 15.0
        s._sweep_direction = -1.0
    elseif s._sweep_hz < 5.0
        s._sweep_direction = 1.0
    s.evs.set_target(s._sweep_hz)
    # Keep sigma_g moderate for exploration
    old_sg = cfg.sigma_g
    cfg.sigma_g = float(clamp(cfg.sigma_g, 0.15, 0.35))
    if cfg.sigma_g != old_sg
        s._log_adaptation("sigma_g", old_sg, cfg.sigma_g, "discovery bounds")
    # Higher learning rate for faster geometry search
    old_lr = cfg.lr_z
    cfg.lr_z = 0.015
    if cfg.lr_z != old_lr
        s._log_adaptation("lr_z", old_lr, cfg.lr_z, "discovery exploration")
end

function _adapt_lock_on(s::AdaptiveAudioEngineState, snap, trend)
    cfg = s.ssgf.cfg
    # If EVS is declining, increase geometry feedback
    if trend < -0.5
        old_sg = cfg.sigma_g
        new_sg = float(clamp(cfg.sigma_g + 0.02, 0.1, 0.6))
        if new_sg != old_sg
            cfg.sigma_g = new_sg
            s._log_adaptation("sigma_g", old_sg, new_sg, "EVS declining, boost coupling")
    # If EVS is improving, reduce learning rate to stabilise
    if trend > 0.5
        old_lr = cfg.lr_z
        new_lr = float(clamp(cfg.lr_z * 0.95, 0.002, 0.02))
        if new_lr != old_lr
            cfg.lr_z = new_lr
            s._log_adaptation("lr_z", old_lr, new_lr, "EVS improving, stabilise")
    # Responsive target adjustment based on peak alignment
    if snap.peak_alignment < 0.5 && snap.peak_hz > 0.5
        # Nudge target toward actual brain peak
        delta = (snap.peak_hz - snap.target_hz) * 0.1
        new_target = float(clamp(snap.target_hz + delta, 0.5, 40.0))
        s.evs.set_target(new_target)
end

function _adapt_deepening(s::AdaptiveAudioEngineState, snap, trend)
    cfg = s.ssgf.cfg
    # Increase field pressure to encourage synchrony
    old_fp = cfg.field_pressure
    new_fp = float(clamp(cfg.field_pressure + 0.005, 0.05, 0.4))
    if new_fp != old_fp
        cfg.field_pressure = new_fp
        s._log_adaptation("field_pressure", old_fp, new_fp, "deepening push")
    # Increase sigma_g gradually
    old_sg = cfg.sigma_g
    new_sg = float(clamp(cfg.sigma_g + 0.005, 0.2, 0.8))
    if new_sg != old_sg
        cfg.sigma_g = new_sg
        s._log_adaptation("sigma_g", old_sg, new_sg, "deepening geometry boost")
    # Lower learning rate for stability
    old_lr = cfg.lr_z
    new_lr = float(clamp(cfg.lr_z * 0.98, 0.001, 0.01))
    if new_lr != old_lr
        cfg.lr_z = new_lr
        s._log_adaptation("lr_z", old_lr, new_lr, "deepening stabilise")
    # If R > 0.9, we're close to theurgic -- fine-tune
    if s.ssgf.R_global > 0.9
        old_fp2 = cfg.field_pressure
        new_fp2 = float(clamp(cfg.field_pressure + 0.01, 0.1, 0.5))
        if new_fp2 != old_fp2
            cfg.field_pressure = new_fp2
            s._log_adaptation("field_pressure", old_fp2, new_fp2, "near-theurgic push")
end

function _log_adaptation(s::AdaptiveAudioEngineState)
    self,
    param: str,
    old: float,
    new: float,
    reason: str,
    ) -> nothing
    record = _AdaptationRecord(
        tick=s._tick,
        phase=s._phase.value,
        param=param,
        old_value=old,
        new_value=new,
        reason=reason,
    )
    s._adaptations = push!(, record)
    logger.debug(
        "Tick %d [%s] %s: %.4f -> %.4f (%s)",
        s._tick,
        s._phase.value,
        param,
        old,
        new,
        reason,
    )
end

function get_session_report(s::AdaptiveAudioEngineState)
    total = length(s._evs_scores)
    avg_evs = float(mean(s._evs_scores)) if s._evs_scores else 0.0
    peak_evs = float(np.max(s._evs_scores)) if s._evs_scores else 0.0
    verified_pct = (s._verified_count / total * 100.0) if total > 0 else 0.0
    # Phase durations
    phase_durations: Dict[str, int] = {}
    if s._tick > 0
        if s._tick <= _DISCOVERY_TICKS
            phase_durations["discovery"] = s._tick
        elseif s._tick <= _LOCKON_TICKS
            phase_durations["discovery"] = _DISCOVERY_TICKS
            phase_durations["lock_on"] = s._tick - _DISCOVERY_TICKS
        else
            phase_durations["discovery"] = _DISCOVERY_TICKS
            phase_durations["lock_on"] = _LOCKON_TICKS - _DISCOVERY_TICKS
            phase_durations["deepening"] = s._tick - _LOCKON_TICKS
    return AdaptiveSessionReport(
        total_ticks=total,
        avg_evs=avg_evs,
        peak_evs=peak_evs,
        verified_pct=verified_pct,
        grade=_compute_grade(verified_pct),
        adaptations=length(s._adaptations),
        phase_durations=phase_durations,
        final_audio=s.ssgf.get_audio_mapping(),
    )
end

function current_phase(s::AdaptiveAudioEngineState)
    return s._phase
end

function tick(s::AdaptiveAudioEngineState)
    return s._tick
end

function reset(s::AdaptiveAudioEngineState)
    s._tick = 0
    s._phase = SessionPhase.DISCOVERY
    s._phase_start_tick = 0
    s._evs_scores.clear()
    s._verified_count = 0
    s._recent_evs.clear()
    s._adaptations.clear()
    s._sweep_direction = 1.0
    s._sweep_hz = 10.0 if s.profile is nothing else s.profile.get_best_target_hz()
end

end # module AdaptiveEngineAccel

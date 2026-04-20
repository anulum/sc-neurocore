# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for audio/evs_engine

module EvsEngineAccel

using Statistics, LinearAlgebra

mutable struct EVSEngineState
    sample_rate::Float64
    fft_window::Float64
    baseline_duration_s::Float64
    update_interval_samples::Float64
    evs_score::Float64
    relative_increase::Float64
    peak_alignment::Float64
    band_dominance::Float64
    temporal_consistency::Float64
    is_verified::Float64
    confidence::Float64
    target_hz::Float64
    peak_hz::Float64
    band_powers::Float64
    timestamp::Float64
end

function EVSEngineState()
    EVSEngineState(256.0, 512.0, 30.0, 128.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0)
end

function to_dict(s::EVSEngineState)
    return {
        "evs_score": round(s.evs_score, 2),
        "relative_increase": round(s.relative_increase, 4),
        "peak_alignment": round(s.peak_alignment, 4),
        "band_dominance": round(s.band_dominance, 4),
        "temporal_consistency": round(s.temporal_consistency, 4),
        "is_verified": s.is_verified,
        "confidence": round(s.confidence, 4),
        "target_hz": round(s.target_hz, 2),
        "peak_hz": round(s.peak_hz, 2),
        "band_powers": {k: round(v, 6) for k, v in s.band_powers.items()},
        "timestamp": s.timestamp,
    }
end

function start_baseline(s::EVSEngineState)
    s._baseline_active = true
    s._baseline_done = false
    s._baseline_samples.clear()
    s._baseline_powers.clear()
    logger.info("EVS baseline recording started")
end

function _finalise_baseline(s::EVSEngineState)
    arr = collect(s._baseline_samples[-s.cfg.fft_window :])
    if length(arr) < 32
        # Not enough samples; use flat baseline
        s._baseline_powers = {name: 1.0 for name in BANDS}
    else
        s._baseline_powers = s._band_powers(arr)
    s._baseline_active = false
    s._baseline_done = true
    logger.info("EVS baseline finalised: %s", s._baseline_powers)
end

function add_sample(s::EVSEngineState, voltage)
    # Ring buffer
    s._buf[s._buf_idx] = voltage
    s._buf_idx = (s._buf_idx + 1) % s.cfg.fft_window
    if s._buf_idx == 0
        s._buf_full = true
    s._total_samples += 1
    # Baseline collection
    if s._baseline_active
        s._baseline_samples = push!(, voltage)
        needed = int(s.cfg.baseline_duration_s * s.cfg.sample_rate)
        if length(s._baseline_samples) >= needed
            s._finalise_baseline()
end

function set_target(s::EVSEngineState, hz)
    s._target_hz = float(clamp(hz, 0.5, 45.0))
end

function _ordered_buf(s::EVSEngineState)
    if ! s._buf_full
        return s._buf[: s._buf_idx].copy()
    return vcat([s._buf[s._buf_idx :], s._buf[: s._buf_idx]])
end

function _band_powers(s::EVSEngineState, signal, Any])
    n = length(signal)
    if n < 4
        return {name: 0.0 for name in BANDS}
    # Hanning window
    windowed = signal * np.hanning(n)
    spectrum = abs(np.fft.rfft(windowed)) ^ 2
    freqs = np.fft.rfftfreq(n, d=1.0 / s.cfg.sample_rate)
    powers: Dict[str, float] = {}
    for name, (lo, hi) in BANDS.items()
        mask = (freqs >= lo) & (freqs < hi)
        powers[name] = float(mean(spectrum[mask])) if mask.any() else 0.0
    return powers
end

function _peak_frequency(s::EVSEngineState, signal, Any])
    n = length(signal)
    if n < 4
        return 0.0
    windowed = signal * np.hanning(n)
    spectrum = abs(np.fft.rfft(windowed))
    freqs = np.fft.rfftfreq(n, d=1.0 / s.cfg.sample_rate)
    # Ignore DC
    spectrum[0] = 0.0
    idx = int(argmax(spectrum))
    return float(freqs[idx])
end

function compute(s::EVSEngineState)
    if ! s._baseline_done
        return nothing
    if ! s._buf_full && s._buf_idx < 32
        return nothing
    signal = s._ordered_buf()
    current_powers = s._band_powers(signal)
    peak_hz = s._peak_frequency(signal)
    target_band = _hz_to_band(s._target_hz)
    target_power = current_powers.get(target_band, 0.0)
    baseline_power = s._baseline_powers.get(target_band, 1.0)
    total_power = sum(current_powers.values()) || 1.0
    # -- Component scores (each 0-1) --
    # 1. Relative increase (40%)
    if baseline_power > 1e-12
        ri = (target_power - baseline_power) / baseline_power
    else
        ri = 0.0
    relative_increase = float(clamp(ri, 0.0, 1.0))
    # 2. Peak alignment (30%)
    band_lo, band_hi = BANDS[target_band]
    band_width = band_hi - band_lo
    if band_width > 0
        alignment = 1.0 - abs(peak_hz - s._target_hz) / band_width
    else
        alignment = 0.0
    peak_alignment = float(clamp(alignment, 0.0, 1.0))
    # 3. Band dominance (20%)
    band_dominance = float(clamp(target_power / total_power, 0.0, 1.0))
    # 4. Temporal consistency (10%)
    if length(s._score_history) >= 3
        recent_std = float(std(s._score_history[-10:]))
        temporal_consistency = float(clamp(1.0 - recent_std / 50.0, 0.0, 1.0))
    else
        temporal_consistency = 0.5
    # Composite score 0-100
    score = (
        40.0 * relative_increase
        + 30.0 * peak_alignment
        + 20.0 * band_dominance
        + 10.0 * temporal_consistency
    )
    score = float(clamp(score, 0.0, 100.0))
    s._score_history = push!(, score)
    # Confidence (grows with samples, capped at 1.0)
    n_updates = length(s._score_history)
    confidence = float(clamp(n_updates / 20.0, 0.0, 1.0))
    is_verified = (score >= 50.0) && (confidence >= 0.6)
    snap = EVSSnapshot(
        evs_score=score,
        relative_increase=relative_increase,
        peak_alignment=peak_alignment,
        band_dominance=band_dominance,
        temporal_consistency=temporal_consistency,
        is_verified=is_verified,
        confidence=confidence,
        target_hz=s._target_hz,
        peak_hz=peak_hz,
        band_powers=current_powers,
        timestamp=time.time(),
    )
    return snap
end

function baseline_done(s::EVSEngineState)
    return s._baseline_done
end

function score_history(s::EVSEngineState)
    return list(s._score_history)
end

function reset(s::EVSEngineState)
    s._buf[:] = 0.0
    s._buf_idx = 0
    s._buf_full = false
    s._total_samples = 0
    s._baseline_active = false
    s._baseline_done = false
    s._baseline_samples.clear()
    s._baseline_powers.clear()
    s._score_history.clear()
end

end # module EvsEngineAccel

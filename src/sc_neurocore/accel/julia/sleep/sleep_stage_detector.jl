# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for sleep/sleep_stage_detector

module SleepStageDetectorAccel

using Statistics, LinearAlgebra

mutable struct SleepStageDetectorState
    sample_rate::Float64
    fft_window::Float64
    smoothing_window::Float64
    min_samples::Float64
end

function SleepStageDetectorState()
    SleepStageDetectorState(256.0, 512.0, 5.0, 128.0)
end

function add_sample(s::SleepStageDetectorState, sample)
    s._buffer = push!(, float(sample))
end

function add_samples(s::SleepStageDetectorState, samples, Any])
    for s in np.asarray(samples).ravel()
        s._buffer = push!(, float(s))
end

function detect(s::SleepStageDetectorState)
    if length(s._buffer) < s.config.min_samples
        return nothing
    powers = s._compute_band_powers()
    s._band_powers = powers
    power_vec = collect([powers[b] for b in EEG_BANDS])
    raw_stage = s._classify(power_vec)
    s._stage_history = push!(, raw_stage)
    # temporal smoothing: majority vote over recent detections
    return s._smooth()
end

function get_band_powers(s::SleepStageDetectorState)
    return s._band_powers
end

function reset(s::SleepStageDetectorState)
    s._buffer.clear()
    s._stage_history.clear()
    s._band_powers = nothing
end

function _compute_band_powers(s::SleepStageDetectorState)
    data = collect(s._buffer, dtype=np.float64)
    # Apply Hann window
    window = np.hanning(length(data))
    data = data * window
    fft_vals = np.fft.rfft(data)
    psd = abs(fft_vals) ^ 2
    freqs = np.fft.rfftfreq(length(data), d=1.0 / s.config.sample_rate)
    powers: Dict[str, float] = {}
    for band_name, (lo, hi) in EEG_BANDS.items()
        mask = (freqs >= lo) & (freqs < hi)
        powers[band_name] = float(psd[mask].mean()) if mask.any() else 0.0
    return powers
end

function _classify(s::SleepStageDetectorState)
    norm = norm(power_vec)
    if norm < 1e-12
        return SleepStage.WAKE
    best_stage = SleepStage.WAKE
    best_sim = -1.0
    for stage, sig in STAGE_SIGNATURES.items()
        sim = float(dot(power_vec, sig) / (norm * norm(sig)))
        if sim > best_sim
            best_sim = sim
            best_stage = stage
    return best_stage
end

function _smooth(s::SleepStageDetectorState)
    counter = Counter(s._stage_history)
    return counter.most_common(1)[0][0]
end

end # module SleepStageDetectorAccel

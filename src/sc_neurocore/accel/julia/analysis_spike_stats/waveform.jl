# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for analysis_spike_stats/waveform

module WaveformAccel

using Statistics, LinearAlgebra

function waveform_width(waveform, dt)
    trough = argmin(waveform)
    if trough >= waveform.size - 1
        return float("nan")
    peak = trough + argmax(waveform[trough:])
    return float((peak - trough) * dt)
end

function waveform_amplitude(waveform)
    return float(np.max(waveform) - np.min(waveform))
end

function waveform_repolarization_slope(waveform, dt)
    trough = argmin(waveform)
    if trough >= waveform.size - 2
        return float("nan")
    post_trough = waveform[trough:]
    dv = diff(post_trough) / dt
    return float(np.max(dv))
end

function waveform_recovery_slope(waveform, dt)
    trough = argmin(waveform)
    if trough >= waveform.size - 1
        return float("nan")
    peak = trough + argmax(waveform[trough:])
    if peak >= waveform.size - 2
        return float("nan")
    post_peak = waveform[peak:]
    dv = diff(post_peak) / dt
    if dv.size == 0
        return float("nan")
    return float(np.min(dv))
end

function waveform_halfwidth(waveform, dt)
    trough_val = np.min(waveform)
    half_val = trough_val / 2.0
    below = findall(waveform < half_val)[0]
    if below.size < 2
        return float("nan")
    return float((below[-1] - below[0]) * dt)
end

function waveform_pt_ratio(waveform)
    trough = argmin(waveform)
    trough_val = abs(waveform[trough])
    if trough >= waveform.size - 1 || trough_val < 1e-30
        return float("nan")
    peak_val = np.max(waveform[trough:])
    return float(abs(peak_val) / trough_val)
end

end # module WaveformAccel

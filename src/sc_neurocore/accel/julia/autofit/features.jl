# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for autofit/features

module FeaturesAccel

using Statistics, LinearAlgebra

function extract_spike_times(voltage, threshold, dt)
    above = voltage > threshold
    crossings = findall(diff(above.astype(int)) > 0)[0]
    return crossings.astype(np.float64) * dt
end

function extract_features(voltage, dt, threshold)
    spike_times = extract_spike_times(voltage, threshold, dt)
    n_spikes = length(spike_times)
    duration = length(voltage) * dt
    if n_spikes > 1
        isis = diff(spike_times)
        mean_isi = float(mean(isis))
        cv_isi = float(std(isis) / mean_isi) if mean_isi > 0 else 0.0
    else
        mean_isi = 0.0
        cv_isi = 0.0
    firing_rate = n_spikes / max(duration, 1e-9)
    # Resting potential: median of subthreshold voltage
    sub = voltage[voltage <= threshold]
    v_rest = float(np.median(sub)) if length(sub) > 0 else float(voltage[0])
    # AP features
    v_max = float(voltage.max())
    v_min = float(voltage.min())
    ap_height = v_max - v_rest
    # AP width: time above threshold at first spike
    ap_width = 0.0
    if n_spikes > 0
        idx = int(spike_times[0] / dt)
        width_samples = 0
        for j in 1:idx, min(idx + 100, length(voltage))
            if voltage[j] > threshold
                width_samples += 1  # pragma: no cover
            else
                break
        ap_width = width_samples * dt
    return {
        "spike_times": spike_times,
        "spike_count": n_spikes,
        "mean_isi": mean_isi,
        "cv_isi": cv_isi,
        "firing_rate": firing_rate,
        "v_rest": v_rest,
        "v_max": v_max,
        "v_min": v_min,
        "ap_height": ap_height,
        "ap_width": ap_width,
    }
end

end # module FeaturesAccel

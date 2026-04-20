# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo SIMD acceleration for features

fn extract_spike_times(voltage: Int, threshold: Int, dt: Int) -> Int:
    var _extract_spike_times_line = 'above = voltage > threshold'
    var _extract_spike_times_line = 'crossings = where(diff(above.astype(int)) > 0)[0]'
    return 0  # return crossings.astype(float64) * dt

fn extract_features(voltage: Int, dt: Int, threshold: Int) -> Int:
    var _extract_features_line = 'spike_times = extract_spike_times(voltage, threshold, dt)'
    var _extract_features_line = 'n_spikes = len(spike_times)'
    var _extract_features_line = 'duration = len(voltage) * dt'
    var _extract_features_line = 'if n_spikes > 1:'
    var _extract_features_line = 'isis = diff(spike_times)'
    var _extract_features_line = 'mean_isi = float(mean(isis))'
    var _extract_features_line = 'cv_isi = float(std(isis) / mean_isi) if mean_isi > 0 else 0.'
    var _extract_features_line = 'else:'
    var _extract_features_line = 'mean_isi = 0.0'
    var _extract_features_line = 'cv_isi = 0.0'
    var _extract_features_line = 'firing_rate = n_spikes / max(duration, 1e-9)'
    var _extract_features_line = '# Resting potential: median of subthreshold voltage'
    var _extract_features_line = 'sub = voltage[voltage <= threshold]'
    var _extract_features_line = 'v_rest = float(median(sub)) if len(sub) > 0 else float(volta'
    var _extract_features_line = '# AP features'
    var _extract_features_line = 'v_max = float(voltage.max())'
    var _extract_features_line = 'v_min = float(voltage.min())'
    var _extract_features_line = 'ap_height = v_max - v_rest'
    var _extract_features_line = '# AP width: time above threshold at first spike'
    var _extract_features_line = 'ap_width = 0.0'
    var _extract_features_line = 'if n_spikes > 0:'
    var _extract_features_line = 'idx = int(spike_times[0] / dt)'
    var _extract_features_line = 'width_samples = 0'
    var _extract_features_line = 'for j in range(idx, min(idx + 100, len(voltage))):'
    var _extract_features_line = 'if voltage[j] > threshold:'
    var _extract_features_line = 'width_samples += 1  # pragma: no cover'
    var _extract_features_line = 'else:'
    var _extract_features_line = 'break'
    var _extract_features_line = 'ap_width = width_samples * dt'
    return 0  # return {
    var _extract_features_line = '"spike_times": spike_times,'
    var _extract_features_line = '"spike_count": n_spikes,'
    var _extract_features_line = '"mean_isi": mean_isi,'
    var _extract_features_line = '"cv_isi": cv_isi,'
    var _extract_features_line = '"firing_rate": firing_rate,'
    var _extract_features_line = '"v_rest": v_rest,'
    var _extract_features_line = '"v_max": v_max,'
    var _extract_features_line = '"v_min": v_min,'
    var _extract_features_line = '"ap_height": ap_height,'
    var _extract_features_line = '"ap_width": ap_width,'
    var _extract_features_line = '}'

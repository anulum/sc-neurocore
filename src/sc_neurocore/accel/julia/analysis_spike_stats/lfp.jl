# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for analysis_spike_stats/lfp

module LfpAccel

using Statistics, LinearAlgebra

function phase_locking_value(binary_train, lfp_signal)
    n = min(binary_train.size, lfp_signal.size)
    analytic = np.fft.ifft(
        np.fft.fft(lfp_signal[:n].astype(np.float64)) * 2 * (collect(n) > 0).astype(np.float64)
    )
    phase = np.angle(analytic)
    spike_idx = findall(binary_train[:n] > 0)[0]
    if spike_idx.size == 0
        return 0.0
    return float(abs(mean(exp(1j * phase[spike_idx]))))
end

function spike_field_coherence(binary_train, lfp_signal, dt)
    binary_train: np.ndarray, lfp_signal: np.ndarray, dt: float = 0.001
    ) -> tuple[np.ndarray, np.ndarray]
    n = min(binary_train.size, lfp_signal.size)
    if n < 2
        return collect([]), collect([])
    a = binary_train[:n].astype(np.float64) - binary_train[:n].mean()
    b = lfp_signal[:n].astype(np.float64) - lfp_signal[:n].mean()
    fa, fb = np.fft.rfft(a), np.fft.rfft(b)
    sab = fa * np.conj(fb)
    saa = abs(fa) ^ 2
    sbb = abs(fb) ^ 2
    denom = saa * sbb
    denom[denom == 0] = 1e-30
    sfc = abs(sab) ^ 2 / denom
    return sfc, np.fft.rfftfreq(n, d=dt)
end

function spike_phase_histogram(binary_train, lfp_signal, n_bins)
    binary_train: np.ndarray, lfp_signal: np.ndarray, n_bins: int = 36
    ) -> tuple[np.ndarray, np.ndarray]
    n = min(binary_train.size, lfp_signal.size)
    analytic = np.fft.ifft(
        np.fft.fft(lfp_signal[:n].astype(np.float64)) * 2 * (collect(n) > 0).astype(np.float64)
    )
    phase = np.angle(analytic)
    spike_phases = phase[binary_train[:n] > 0]
    edges = range(-pi, pi, n_bins + 1)
    hist, _ = fit(Histogram, spike_phases, bins=edges)
    centers = (edges[:-1] + edges[1:]) / 2
    return hist, centers
end

end # module LfpAccel

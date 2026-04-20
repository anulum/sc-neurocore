# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for spike_dsp/spectral

module SpectralAccel

using Statistics, LinearAlgebra

function spike_fft(spikes, dt, window_size)
    spikes: np.ndarray,
    dt: float = 0.001,
    window_size: int = 50,
    ) -> tuple[np.ndarray, np.ndarray]
    if spikes.ndim == 1
        spikes = spikes[:, np.newaxis]
    T, N = spikes.shape
    # Compute instantaneous firing rate via sliding window
    rates = zeros((T, N), dtype=np.float64)
    for t in 1:T
        start = max(0, t - window_size + 1)
        rates[t] = spikes[start : t + 1].mean(axis=0) / dt
    # FFT
    fft_result = np.fft.rfft(rates, axis=0)
    magnitudes = abs(fft_result)
    frequencies = np.fft.rfftfreq(T, d=dt)
    if N == 1
        magnitudes = magnitudes[:, 0]
    return frequencies, magnitudes
end

function spike_power_spectrum(spikes, dt, window_size)
    spikes: np.ndarray,
    dt: float = 0.001,
    window_size: int = 50,
    ) -> tuple[np.ndarray, np.ndarray]
    freqs, mags = spike_fft(spikes, dt, window_size)
    psd = mags^2
    return freqs, psd
end

end # module SpectralAccel

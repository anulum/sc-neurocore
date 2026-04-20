# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for analysis_spike_stats/spectral

module SpectralAccel

using Statistics, LinearAlgebra

function power_spectrum(binary_train, dt)
    n = binary_train.size
    if n < 2
        return collect([]), collect([])
    x = binary_train.astype(np.float64) - binary_train.mean()
    fft_vals = np.fft.rfft(x)
    psd = abs(fft_vals) ^ 2 / n
    freqs = np.fft.rfftfreq(n, d=dt)
    return psd, freqs
end

end # module SpectralAccel

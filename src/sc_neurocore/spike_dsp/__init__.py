# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Spike-domain digital signal processing

"""Spike-domain DSP: FIR/IIR filters, FFT, wavelet decomposition in spike domain."""

from .filters import SpikeFIR, SpikeIIR, spike_convolve
from .spectral import spike_fft, spike_power_spectrum
from .wavelets import spike_wavelet_decompose

__all__ = [
    "SpikeFIR",
    "SpikeIIR",
    "spike_convolve",
    "spike_fft",
    "spike_power_spectrum",
    "spike_wavelet_decompose",
]

# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Bitstream-to-optical conversion

"""Convert stochastic bitstreams into typed optical pulse representations."""

from __future__ import annotations

import math
from typing import Any, List

import numpy as np
import numpy.typing as npt

from ._photonic_types import (
    OpticalModulation,
    OpticalPulse,
    PhotonicTarget,
    _require_non_negative,
    _require_positive,
)


def _normalise_bitstream(bitstream: np.ndarray[Any, Any]) -> npt.NDArray[np.float64]:
    """Return a one-dimensional finite binary float64 array."""
    array = np.asarray(bitstream)
    if array.ndim != 1:
        raise ValueError(f"bitstream must be one-dimensional, got shape {array.shape}")
    try:
        normalised = array.astype(np.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError("bitstream must contain numeric binary values") from exc
    if not np.all(np.isfinite(normalised)):
        raise ValueError("bitstream must contain only finite values")
    if not np.all((normalised == 0.0) | (normalised == 1.0)):
        raise ValueError("bitstream values must be exactly 0 or 1")
    return normalised


class BitstreamToOptical:
    """Convert SC bitstreams into optical pulse trains."""

    def __init__(self, target: PhotonicTarget):
        if not isinstance(target, PhotonicTarget):
            raise TypeError("target must be a PhotonicTarget")
        self.target = target

    def convert(
        self,
        bitstream: np.ndarray[Any, Any],
        pulse_duration_ps: float = 10.0,
    ) -> List[OpticalPulse]:
        """Map a binary SC bitstream to an optical pulse train.

        Phase modulation maps one to phase zero and zero to phase π.
        Amplitude modulation maps one to unit amplitude and zero to zero.
        Hybrid modulation combines phase and amplitude encoding.
        """
        _require_positive(pulse_duration_ps, "pulse_duration_ps")
        pulses: List[OpticalPulse] = []
        for bit in _normalise_bitstream(bitstream):
            b = int(bit)
            if self.target.modulation == OpticalModulation.PHASE:
                phase = 0.0 if b else math.pi
                amplitude = 1.0
            elif self.target.modulation == OpticalModulation.AMPLITUDE:
                phase = 0.0
                amplitude = float(b)
            else:
                phase = 0.0 if b else math.pi / 2
                amplitude = 0.8 + 0.2 * float(b)
            pulses.append(
                OpticalPulse(
                    phase=phase,
                    amplitude=amplitude,
                    wavelength_nm=self.target.wavelength_nm,
                    duration_ps=pulse_duration_ps,
                )
            )
        return pulses

    def to_phase_array(self, bitstream: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Return the vectorised phase encoding in radians."""
        bs = _normalise_bitstream(bitstream)
        if self.target.modulation == OpticalModulation.PHASE:
            return np.where(bs > 0.5, 0.0, math.pi)
        if self.target.modulation == OpticalModulation.AMPLITUDE:
            return np.zeros_like(bs)
        return np.where(bs > 0.5, 0.0, math.pi / 2)

    def to_amplitude_array(self, bitstream: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Return the vectorised normalised amplitude encoding."""
        bs = _normalise_bitstream(bitstream)
        if self.target.modulation == OpticalModulation.PHASE:
            return np.ones_like(bs)
        if self.target.modulation == OpticalModulation.AMPLITUDE:
            return bs
        return 0.8 + 0.2 * bs

    def optical_power_profile(
        self,
        bitstream: np.ndarray[Any, Any],
        input_power_mw: float = 1.0,
    ) -> np.ndarray[Any, Any]:
        """Compute output power after the target insertion loss."""
        _require_non_negative(input_power_mw, "input_power_mw")
        amplitudes = self.to_amplitude_array(bitstream)
        loss_linear = 10.0 ** (-self.target.insertion_loss_db / 10.0)
        optical_power: np.ndarray[Any, Any] = amplitudes * amplitudes * input_power_mw * loss_linear
        return optical_power


__all__ = ["BitstreamToOptical"]

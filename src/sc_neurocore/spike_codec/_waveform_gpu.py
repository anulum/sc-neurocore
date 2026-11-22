# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — The GPU encoder of the lossless waveform stream

"""Open the engine's GPU encoder for the lossless waveform stream.

The encoder exists only in an engine built with its ``gpu`` feature (wgpu), and it runs on a
physical GPU adapter only: the engine refuses a software adapter with the reason. This module
holds everything that runs only after that construction succeeds, so a host without a physical
GPU reaches none of it.
"""

from __future__ import annotations

from types import ModuleType
from typing import Any


def open_gpu_encoder(
    engine: ModuleType, channels: int, adc_bits: int, capacity: int
) -> tuple[Any, tuple[str, str], str]:
    """Return the engine's GPU encoder, the adapter it runs on, and the backend name.

    Parameters
    ----------
    engine : ModuleType
        The engine extension; the caller has checked that it exports
        ``WaveformLosslessGpuEncoder``.
    channels : int
        The number of channels the encoder takes.
    adc_bits : int
        The converter's resolution in bits.
    capacity : int
        The most samples the encoder accepts in one call.

    Returns
    -------
    tuple
        The encoder, its adapter as ``(device_name, device_api)``, and ``"gpu"``.
    """
    coder = engine.WaveformLosslessGpuEncoder(channels, adc_bits, capacity)
    return coder, (coder.device_name, coder.device_api), "gpu"


__all__ = ["open_gpu_encoder"]

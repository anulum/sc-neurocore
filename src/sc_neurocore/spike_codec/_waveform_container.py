# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Python bridge to the complete native waveform codec

"""Locate the complete native API and pass its options across the extension boundary."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from types import ModuleType
from typing import Any, Protocol, cast

import numpy as np

NATIVE_CONTAINER_SYMBOLS = ("waveform_container_encode", "waveform_container_decode")


@dataclass(frozen=True, slots=True)
class NativeWaveformOptions:
    """Validated options passed as attributes to the complete native encoder."""

    mode: str
    threshold_sigma: float
    snippet_samples: int
    max_templates: int
    template_threshold: float
    quantize_bits: int
    adc_bits: int | None
    zstandard: bool
    denoise_background: bool


class NativeWaveformCodec(Protocol):
    """Whole-container functions exposed by the installed Rust extension."""

    WAVEFORM_CONTAINER_ZSTD: bool

    def waveform_container_encode(
        self, samples: np.ndarray[Any, Any], options: NativeWaveformOptions
    ) -> tuple[bytes, int, int, int, int, int]:
        """Encode and return the packet, event counts and section lengths."""
        ...

    def waveform_container_decode(
        self, packet: bytes, max_samples: int
    ) -> tuple[str, np.ndarray[Any, Any] | None, np.ndarray[Any, Any] | None]:
        """Return the mode, detected raster and reconstructed samples."""
        ...


def native_container(engine: ModuleType | None, backend: str) -> NativeWaveformCodec | None:
    """Resolve native operations, requiring the complete API in explicit Rust mode."""
    if engine is not None and all(
        callable(getattr(engine, symbol, None)) for symbol in NATIVE_CONTAINER_SYMBOLS
    ):
        return cast(NativeWaveformCodec, engine)
    if backend == "rust":
        raise ValueError("backend 'rust' needs sc_neurocore_engine with the complete WaveformCodec")
    return None


def has_optional_module(name: str) -> bool:
    """Check an optional dependency and propagate errors from a present installation."""
    try:
        importlib.import_module(name)
    except ModuleNotFoundError as error:
        if error.name != name:
            raise
        return False
    return True

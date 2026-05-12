# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Spike train compression codec

"""Neural data compression: spike rasters (50-750x) and raw waveforms (10-24x)."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

from .codec import SpikeCodec, CompressionResult
from .delta_codec import DeltaSpikeCodec, DeltaCompressionResult
from .streaming_codec import StreamingSpikeCodec, StreamingCompressionResult
from .aer_codec import AERSpikeCodec, AERCompressionResult
from .waveform_codec import WaveformCodec, WaveformCompressionResult
from .registry import get_codec, list_codecs, recommend_codec

_LAZY_EXPORTS = {
    "PredictiveSpikeCodec": (".predictive_codec", "PredictiveSpikeCodec"),
    "PredictiveCompressionResult": (".predictive_codec", "PredictiveCompressionResult"),
}


def __getattr__(name: str) -> Any:
    if name in _LAZY_EXPORTS:
        module_name, symbol_name = _LAZY_EXPORTS[name]
        module = import_module(module_name, __name__)
        value = getattr(module, symbol_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if TYPE_CHECKING:
    from .predictive_codec import PredictiveCompressionResult, PredictiveSpikeCodec

__all__ = [
    "SpikeCodec",
    "CompressionResult",
    "PredictiveSpikeCodec",
    "PredictiveCompressionResult",
    "DeltaSpikeCodec",
    "DeltaCompressionResult",
    "StreamingSpikeCodec",
    "StreamingCompressionResult",
    "AERSpikeCodec",
    "AERCompressionResult",
    "WaveformCodec",
    "WaveformCompressionResult",
    "get_codec",
    "list_codecs",
    "recommend_codec",
]

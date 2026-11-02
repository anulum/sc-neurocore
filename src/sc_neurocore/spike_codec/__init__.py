# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Spike train compression codec

"""Neural data compression: spike-raster codecs and a raw-waveform codec with lossy and lossless modes."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

from .codec import SpikeCodec, CompressionResult
from .delta_codec import DeltaSpikeCodec, DeltaCompressionResult
from .streaming_codec import StreamingSpikeCodec, StreamingCompressionResult
from .aer_codec import AERSpikeCodec, AERCompressionResult
from .waveform_codec import WaveformCodec, WaveformCompressionResult, WaveformDecodeResult
from .waveform_stream import LosslessStreamDecoder, LosslessStreamEncoder

_LAZY_EXPORTS = {
    "CodecWireFormat": (".catalogue_types", "CodecWireFormat"),
    "CodecVariant": (".catalogue_types", "CodecVariant"),
    "CodecDescriptor": (".catalogue_types", "CodecDescriptor"),
    "CodecAvailability": (".catalogue_types", "CodecAvailability"),
    "probe_codec": (".catalogue", "probe_codec"),
    "list_codec_descriptors": (".catalogue", "list_codec_descriptors"),
    "describe_codec": (".catalogue", "describe_codec"),
    "PredictiveSpikeCodec": (".predictive_codec", "PredictiveSpikeCodec"),
    "PredictiveCompressionResult": (".predictive_codec", "PredictiveCompressionResult"),
    "get_codec": (".registry", "get_codec"),
    "list_codecs": (".registry", "list_codecs"),
    "recommend_codec": (".registry", "recommend_codec"),
}


def __getattr__(name: str) -> Any:
    """Resolve a public export without loading unrelated optional backends.

    Parameters
    ----------
    name : str
        Public attribute requested from the codec package.

    Returns
    -------
    object
        The original class or function, cached in the package namespace.

    Raises
    ------
    AttributeError
        If the requested name is not a public lazy export.
    """
    if name in _LAZY_EXPORTS:
        module_name, symbol_name = _LAZY_EXPORTS[name]
        module = import_module(module_name, __name__)
        value = getattr(module, symbol_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if TYPE_CHECKING:
    from .catalogue_types import CodecAvailability, CodecDescriptor, CodecVariant, CodecWireFormat
    from .catalogue import describe_codec, list_codec_descriptors, probe_codec
    from .predictive_codec import PredictiveCompressionResult, PredictiveSpikeCodec
    from .registry import get_codec, list_codecs, recommend_codec

__all__ = [
    "CodecWireFormat",
    "CodecVariant",
    "CodecDescriptor",
    "CodecAvailability",
    "probe_codec",
    "list_codec_descriptors",
    "describe_codec",
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
    "WaveformDecodeResult",
    "LosslessStreamEncoder",
    "LosslessStreamDecoder",
    "get_codec",
    "list_codecs",
    "recommend_codec",
]

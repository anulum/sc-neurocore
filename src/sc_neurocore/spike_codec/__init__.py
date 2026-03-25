# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Spike train compression codec

"""Neural data compression: spike rasters (50-750x) and raw waveforms (10-24x)."""

from .codec import SpikeCodec, CompressionResult
from .predictive_codec import PredictiveSpikeCodec, PredictiveCompressionResult
from .delta_codec import DeltaSpikeCodec, DeltaCompressionResult
from .streaming_codec import StreamingSpikeCodec, StreamingCompressionResult
from .aer_codec import AERSpikeCodec, AERCompressionResult
from .waveform_codec import WaveformCodec, WaveformCompressionResult
from .registry import get_codec, list_codecs, recommend_codec

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

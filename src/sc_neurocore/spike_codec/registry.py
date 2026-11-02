# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Spike codec registry and unified API

"""Codec registry: lookup by name, recommend by data characteristics.

Five raster codecs in this legacy factory:

    isi         — ISI with auto-selected varint/Huffman entropy.
    predictive  — Predictor + XOR residuals.
    delta       — Inter-channel XOR residuals.
    streaming   — Independent frames; window size is in samples.
    aer         — Address-event raster packets.

All share the same API: compress(spikes) → (bytes, result),
decompress(bytes, T, N) → spikes.

Use list_codec_descriptors from sc_neurocore.spike_codec for the full
catalogue, including waveform and stateful stream interfaces. These have
different contracts and are not inserted into this raster-only factory.
"""

from __future__ import annotations

from typing import Any


from .codec import SpikeCodec
from .predictive_codec import PredictiveSpikeCodec
from .delta_codec import DeltaSpikeCodec
from .streaming_codec import StreamingSpikeCodec
from .aer_codec import AERSpikeCodec


CODEC_REGISTRY: dict[str, type] = {
    "isi": SpikeCodec,
    "predictive": PredictiveSpikeCodec,
    "delta": DeltaSpikeCodec,
    "streaming": StreamingSpikeCodec,
    "aer": AERSpikeCodec,
}


def get_codec(name: str, **kwargs: Any) -> Any:
    """Get a codec by name.

    Parameters
    ----------
    name : str
        One of: 'isi', 'predictive', 'delta', 'streaming', 'aer'.
    **kwargs
        Passed to the codec constructor.

    Returns
    -------
    Codec instance with compress/decompress methods.
    """
    cls = CODEC_REGISTRY.get(name)
    if cls is None:
        available = ", ".join(sorted(CODEC_REGISTRY))
        raise ValueError(f"Unknown codec {name!r}. Available: {available}")
    return cls(**kwargs)


def list_codecs() -> list[str]:
    """List available codec names."""
    return sorted(CODEC_REGISTRY)


def recommend_codec(
    n_channels: int,
    firing_rate: float,
    latency_ms: float = 10.0,
    correlated: bool = False,
    neuromorphic: bool = False,
) -> str:
    """Suggest a raster codec using a deterministic configuration heuristic.

    Parameters
    ----------
    n_channels : int
        Number of recording channels.
    firing_rate : float
        Mean firing rate in Hz (per neuron).
    latency_ms : float
        Maximum acceptable latency in milliseconds.
    correlated : bool
        True if nearby channels are spatially correlated.
    neuromorphic : bool
        Request address-event raster coding; this does not verify hardware or UDP compatibility.

    Returns
    -------
    str
        Raster codec name. Priority: neuromorphic -> aer; latency_ms <= 1 ->
        streaming; correlated and channels >= 16 -> delta; channels >= 64 ->
        predictive; otherwise isi. firing_rate is retained for compatibility
        and does not affect this heuristic. No measured speed or latency is promised.
    """
    if neuromorphic:
        return "aer"

    if latency_ms <= 1.0:
        return "streaming"

    if correlated and n_channels >= 16:
        return "delta"

    # Predictive works best when temporal structure exists
    # (periodic bursting, oscillations, drift)
    if n_channels >= 64:
        return "predictive"

    return "isi"

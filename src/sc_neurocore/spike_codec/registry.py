# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Spike codec registry and unified API

"""Codec registry: lookup by name, recommend by data characteristics.

Five codecs for different use cases:

    isi         — Baseline ISI + varint. Simple, general-purpose.
    predictive  — EMA predictor + XOR errors. Best for BCI implants.
    delta       — Inter-channel XOR. Best for correlated probe arrays.
    streaming   — Fixed-latency frames. Best for real-time decoding.
    aer         — Event list. Best for neuromorphic inter-chip routing.

All share the same API: compress(spikes) → (bytes, result),
decompress(bytes, T, N) → spikes.
"""

from __future__ import annotations

import numpy as np

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


def get_codec(name: str, **kwargs):
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
    """Recommend a codec based on data characteristics.

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
        True if target is neuromorphic hardware (Loihi, SpiNNaker).

    Returns
    -------
    str — codec name
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

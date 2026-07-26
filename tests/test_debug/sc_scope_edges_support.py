# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Shared fixtures for SC scope edge tests

"""Shared samples, sessions, and SCC dispatch fixtures for scope edge tests."""

from __future__ import annotations

import numpy as np

from sc_neurocore.debug.sc_scope import (
    BitstreamSample,
    LiveAnalyzer,
    ScopeSession,
    TransportBackend,
    TransportConfig,
    TransportType,
)


def _sample(layer_id: int) -> BitstreamSample:
    """A small single-word bitstream sample for a given layer."""
    return BitstreamSample(
        timestamp_ns=0,
        layer_id=layer_id,
        neuron_id=0,
        words=np.array([0x0F0F0F0F], dtype=np.uint32),
        sample_index=0,
    )


def _session() -> ScopeSession:
    backend = TransportBackend(TransportConfig(TransportType.SIMULATED))
    return ScopeSession(transport=backend, analyzer=LiveAnalyzer(num_layers=2))

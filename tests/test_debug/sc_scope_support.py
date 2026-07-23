# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_sc_scope.py

from __future__ import annotations

import numpy as np
from sc_neurocore.debug.sc_scope import (
    AnalysisWindow,
    BitstreamSample,
    LayerErrorBudget,
    LiveAnalyzer,
    ScopeRenderer,
    ScopeSession,
    TransportBackend,
    TransportConfig,
    TransportType,
    TriggerCondition,
    TriggerEngine,
    TriggerType,
    compute_scc,
)
def _sample(layer: int = 0, density: float = 0.5, n_words: int = 8) -> BitstreamSample:
    rng = np.random.default_rng(42 + layer)
    threshold = int(density * 0xFFFF_FFFF)
    words = rng.integers(0, 0xFFFF_FFFF, size=n_words, dtype=np.uint32)
    packed = np.where(words < threshold, np.uint32(0xFFFF_FFFF), np.uint32(0))
    return BitstreamSample(
        timestamp_ns=layer * 1000,
        layer_id=layer,
        neuron_id=0,
        words=packed,
    )

__all__ = ['np', 'AnalysisWindow', 'BitstreamSample', 'LayerErrorBudget', 'LiveAnalyzer', 'ScopeRenderer', 'ScopeSession', 'TransportBackend', 'TransportConfig', 'TransportType', 'TriggerCondition', 'TriggerEngine', 'TriggerType', 'compute_scc', '_sample']

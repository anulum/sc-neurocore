# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_spike_gnn.py

from __future__ import annotations

from typing import Any
import numpy as np
from sc_neurocore.spike_gnn import SpikeGNNLayer, SpikeGraphConv
def _triangle_graph() -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    adj = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=np.float64)
    features = np.random.rand(3, 8)
    return features, adj

__all__ = ['Any', 'np', 'SpikeGNNLayer', 'SpikeGraphConv', '_triangle_graph']

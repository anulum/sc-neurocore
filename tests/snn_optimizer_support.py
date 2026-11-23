# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_snn_optimizer.py

"""Shared three-layer graph builder for the SNN optimiser suites."""

from __future__ import annotations

import numpy as np
from sc_neurocore.snn_optimizer import (
    SNNGraph,
    LayerNode,
)


def _make_graph() -> SNNGraph:
    """Build a random three-layer graph with low firing rates."""
    return SNNGraph(
        layers=[
            LayerNode("h1", 10, 8, np.random.randn(8, 10), firing_rates=np.full(8, 0.15)),
            LayerNode("h2", 8, 4, np.random.randn(4, 8), firing_rates=np.full(4, 0.1)),
            LayerNode("out", 4, 2, np.random.randn(2, 4), firing_rates=np.full(2, 0.2)),
        ]
    )


__all__ = [
    "_make_graph",
]

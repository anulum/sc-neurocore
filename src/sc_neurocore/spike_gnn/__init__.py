# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Spike-native graph neural networks

"""Spike-based GNN: message passing with spike trains instead of float vectors."""

from .spike_gnn import SpikeGNNLayer, SpikeGraphConv

__all__ = ["SpikeGNNLayer", "SpikeGraphConv"]

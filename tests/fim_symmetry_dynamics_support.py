# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_fim_symmetry_dynamics.py

"""Shared names, a recurrent network builder and a symmetry measure for the FIM tests."""

from __future__ import annotations

import numpy as np
from sc_neurocore import StochasticLIFNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.projection import Projection
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput


def _make_self_connected_network(
    n: int = 30, w: float = 0.3, p: float = 0.3, fim_lambda: float = 0.0
) -> tuple[Network, Projection, SpikeMonitor]:
    """Build a recurrent excitatory population with self-projection."""
    pop = Population(StochasticLIFNeuron, n=n, label="exc")
    proj = Projection(pop, pop, weight=w, probability=p, plasticity="stdp", seed=42)
    drive = PoissonInput(n=n, rate_hz=80.0, weight=2.0, dt=0.001, seed=42)
    mon = SpikeMonitor(pop, label="spk")
    net = Network(pop, proj, drive, mon, fim_lambda=fim_lambda)
    return net, proj, mon


def _symmetry_measure(proj: Projection) -> float:
    """Measure weight matrix asymmetry: ||W - W^T|| / ||W||."""
    n = proj.source.n
    # Build dense matrix from CSR
    W = np.zeros((n, n))
    for i in range(n):
        for k in range(proj.indptr[i], proj.indptr[i + 1]):
            j = proj.indices[k]
            W[i, j] = proj.data[k]
    asym = np.linalg.norm(W - W.T)
    total = np.linalg.norm(W)
    if total < 1e-12:
        return 0.0
    return float(asym / total)


__all__ = [
    "np",
    "StochasticLIFNeuron",
    "Population",
    "Projection",
    "Network",
    "SpikeMonitor",
    "PoissonInput",
    "_make_self_connected_network",
    "_symmetry_measure",
]

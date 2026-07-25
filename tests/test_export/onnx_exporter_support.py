# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SCOnnxExporter contract fixtures

from __future__ import annotations

import os

import numpy as np

from sc_neurocore.layers.vectorized_layer import VectorizedSCLayer


def perf_enabled() -> bool:
    """Return whether wall-clock contract tests were explicitly enabled."""

    return os.environ.get("SC_NEUROCORE_PERF") == "1"


class DummyLayer:
    """Simple layer without a dense class name for op-type testing."""

    def __init__(self, n_inputs: int) -> None:
        self.n_inputs = n_inputs
        self.n_neurons = 3
        self.length = 8


class BareLayer:
    """Layer exposing only the required input count."""

    def __init__(self, n_inputs: int) -> None:
        self.n_inputs = n_inputs


class DenseWeightedLayer:
    """Dense-named layer exposing weights for weight-emitting branches."""

    def __init__(self, n_inputs: int, n_neurons: int) -> None:
        self.n_inputs = n_inputs
        self.n_neurons = n_neurons
        self.length = 8
        self.weights = np.ones((n_inputs, n_neurons), dtype=np.float32)


def make_layers() -> list[VectorizedSCLayer]:
    """Build the deterministic two-layer export fixture."""

    np.random.seed(0)
    layer1 = VectorizedSCLayer(n_inputs=3, n_neurons=2, length=8)
    layer2 = VectorizedSCLayer(n_inputs=2, n_neurons=1, length=8)
    return [layer1, layer2]

# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Shared quantum-annealing test builders

"""Typed deterministic fixtures for quantum-annealing tests."""

from __future__ import annotations

from itertools import product
from typing import Any

import numpy as np
from numpy.typing import NDArray

from sc_neurocore.bridges.quantum_annealing import IsingModel


def unsafe(value: object) -> Any:
    """Expose a deliberate invalid runtime value to boundary tests."""
    return value


def simple_adjacency() -> NDArray[np.float64]:
    """Return a symmetric three-node excitatory chain."""
    return np.array(
        [
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )


def simple_ising() -> IsingModel:
    """Return a labeled three-qubit Ising model."""
    return IsingModel(
        h={0: 0.1, 1: -0.2, 2: 0.0},
        J={(0, 1): -1.0, (1, 2): 0.5},
        qubit_labels={0: "A", 1: "B", 2: "C"},
        n_qubits=3,
        source="test",
    )


def spin_assignments(size: int) -> list[dict[int, int]]:
    """Enumerate all spin assignments for a small model."""
    return [dict(enumerate(values)) for values in product((-1, 1), repeat=size)]

# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Canonical SCPN physical constants

"""
Canonical SCPN physical constants.

Single source of truth for natural frequencies and coupling parameters.
Rust engine reads from engine/src/scpn/params.json (generated from this file).
"""

from __future__ import annotations

from typing import Any

import numpy as np

# Natural frequencies Omega_n (rad/s) for the 16 SCPN layers.
# Source: parameter_catalogue_full.yaml (validated 2025-12-25)
# Canonical reference: SCPN-CODEBASE/optimizations/scpn_params.py
OMEGA_N = np.array(
    [
        1.329,  # L1  Quantum Biological
        251.327,  # L2  Neurochemical (~40 Hz x 2pi)
        0.628,  # L3  Genomic-Epigenetic
        31.416,  # L4  Cellular Synchrony (~5 Hz x 2pi)
        6.283,  # L5  Intentional Frame (~1 Hz x 2pi)
        49.199,  # L6  Gaian / Schumann (~7.83 Hz x 2pi)
        3.142,  # L7  Geometrical-Symbolic
        0.105,  # L8  Cosmic Information
        1.571,  # L9  Memory Manifold
        0.942,  # L10 Identity Manifold
        0.209,  # L11 Noospheric-Cultural
        0.042,  # L12 Ecological-Gaian
        0.013,  # L13 Source-Field
        0.006,  # L14 Transdimensional
        0.003,  # L15 Consilium
        0.991,  # L16 The Director
    ],
    dtype=np.float64,
)

N_LAYERS = 16

# Knm coupling matrix parameters.
# Source: HolonomicAtlas/src/knm_tools/knm_matrix_calculator.py
K_BASE = 0.45
DECAY_ALPHA = 0.3

# Calibration anchors (1-indexed layer pairs -> value)
CALIBRATION_ANCHORS = {
    (1, 2): 0.302,
    (2, 3): 0.201,
    (3, 4): 0.252,
    (4, 5): 0.154,
}

# Cross-hierarchy boosts (1-indexed)
CROSS_BOOSTS = {
    (1, 16): 0.05,  # L1<->L16 quantum-director bridge
    (5, 7): 0.15,  # L5<->L7  intentional-symbolic bridge
}


def build_knm_matrix(n_layers: int = N_LAYERS) -> np.ndarray[Any, Any]:
    """
    Build the Knm inter-layer coupling matrix.

    Construction: exponential decay baseline, calibration anchor overrides,
    cross-hierarchy boosts, symmetrisation, zero diagonal.
    """
    if not isinstance(n_layers, int) or isinstance(n_layers, bool) or n_layers <= 0:
        raise ValueError("n_layers must be a positive integer")

    K = np.zeros((n_layers, n_layers), dtype=np.float64)

    for n in range(n_layers):
        for m in range(n_layers):
            if n != m:
                K[n, m] = K_BASE * np.exp(-DECAY_ALPHA * abs(n - m))

    for (i, j), val in CALIBRATION_ANCHORS.items():
        if i <= n_layers and j <= n_layers:
            K[i - 1, j - 1] = val
            K[j - 1, i - 1] = val

    for (i, j), val in CROSS_BOOSTS.items():
        if i <= n_layers and j <= n_layers:
            K[i - 1, j - 1] = val
            K[j - 1, i - 1] = val

    K[...] = 0.5 * (K + K.T)
    np.fill_diagonal(K, 0.0)

    return K

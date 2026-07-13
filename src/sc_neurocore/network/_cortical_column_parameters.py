# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Published Potjans cortical-column parameters

"""Published parameters and population metadata for the cortical-column model."""

from __future__ import annotations

from typing import Any

import numpy as np

# ── Population ordering and per-population sizes (Potjans Table 5) ──

POPULATIONS: tuple[str, ...] = (
    "L23e",
    "L23i",
    "L4e",
    "L4i",
    "L5e",
    "L5i",
    "L6e",
    "L6i",
)
N_POPS = len(POPULATIONS)

FULL_SIZES: dict[str, int] = {
    "L23e": 20683,
    "L23i": 5834,
    "L4e": 21915,
    "L4i": 5479,
    "L5e": 4850,
    "L5i": 1065,
    "L6e": 14395,
    "L6i": 2948,
}

# K_bg: number of independent background-Poisson channels per cell.
# Source: Potjans & Diesmann 2014 Table 5 column "k_ext".
K_BG: dict[str, int] = {
    "L23e": 1600,
    "L23i": 1500,
    "L4e": 2100,
    "L4i": 1900,
    "L5e": 2000,
    "L5i": 1900,
    "L6e": 2900,
    "L6i": 2100,
}

# Connection-probability matrix. Rows = TARGET, columns = SOURCE.
# Source: Potjans & Diesmann 2014 Table 5 (transcribed verbatim,
# Binzegger et al. 2004 anatomical estimate). Values not in the
# table are 0.
#
# Row order follows POPULATIONS; column order follows POPULATIONS.
CONN_PROBS: np.ndarray[Any, Any] = np.array(
    [
        # src:  L23e    L23i    L4e     L4i     L5e     L5i     L6e     L6i
        [0.1009, 0.1689, 0.0437, 0.0818, 0.0323, 0.0000, 0.0076, 0.0000],  # L23e
        [0.1346, 0.1371, 0.0316, 0.0515, 0.0755, 0.0000, 0.0042, 0.0000],  # L23i
        [0.0077, 0.0059, 0.0497, 0.1350, 0.0067, 0.0003, 0.0453, 0.0000],  # L4e
        [0.0691, 0.0029, 0.0794, 0.1597, 0.0033, 0.0000, 0.1057, 0.0000],  # L4i
        [0.1004, 0.0622, 0.0505, 0.0057, 0.0831, 0.3726, 0.0204, 0.0000],  # L5e
        [0.0548, 0.0269, 0.0257, 0.0022, 0.0598, 0.3158, 0.0086, 0.0000],  # L5i
        [0.0156, 0.0066, 0.0211, 0.0166, 0.0572, 0.0197, 0.0396, 0.2252],  # L6e
        [0.0364, 0.0010, 0.0034, 0.0005, 0.0277, 0.0080, 0.0658, 0.1443],  # L6i
    ],
    dtype=np.float64,
)


# ── LIF + synapse + delay parameters (Potjans Table 5) ──────────────

C_M = 250.0  # pF — membrane capacitance
TAU_M = 10.0  # ms — membrane time constant
TAU_SYN = 0.5  # ms — exponential PSC decay
T_REF = 2.0  # ms — absolute refractory
E_L = -65.0  # mV — leak reversal == reset
V_RESET = -65.0  # mV
V_TH = -50.0  # mV — spike threshold

# Synaptic weights (PSC peak amplitudes, pA). Excitatory mean is
# w; inhibitory weights are −g·w. The L4e → L23e edge is boosted
# to 2·w per Potjans 2014.
W_E = 87.81  # pA
G_INH = 4.0
W_I = -G_INH * W_E

# Synaptic delays (ms). Per Potjans Table 5: per-connection
# Gaussian distributions. Mean + std per source-type. The mean
# values (1.5 / 0.8 ms) are also the legacy "single delay" values
# used when `delay_distribution=False`.
DELAY_E = 1.5
DELAY_E_SIGMA = 0.75
DELAY_I = 0.8
DELAY_I_SIGMA = 0.4

# Background Poisson rate per channel (Hz).
BG_RATE = 8.0


def _is_inhibitory(pop_name: str) -> bool:
    """Return whether ``pop_name`` denotes an inhibitory population."""
    return pop_name.endswith("i")


def population_sizes(scale: float) -> dict[str, int]:
    """Return Potjans population sizes at ``scale`` without building connectivity.

    The full published column has roughly 77k neurons and hundreds of
    millions of synapses. Size contracts must therefore be observable without
    materialising the full synapse graph.
    """
    if not (0.0 < scale <= 1.0):
        raise ValueError(f"scale must be in (0, 1], got {scale}")
    return {pop: max(1, int(round(FULL_SIZES[pop] * scale))) for pop in POPULATIONS}

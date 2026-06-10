# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Storage strategy recommender

"""Determine optimal storage strategy (Registers/BRAM/URAM) for neuron state."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class StorageRecommendation:
    """Storage recommendation for neuron array state.

    Attributes
    ----------
    strategy : str
        ``"registers"``, ``"bram"``, or ``"uram"``.
    neuron_count : int
        Number of neurons in the array.
    total_bits : int
        Total state bits.
    bram_18k_used : int
        Estimated 18Kb BRAM tiles consumed.
    bram_36k_used : int
        Estimated 36Kb BRAM tiles consumed.
    uram_used : int
        Estimated URAM tiles consumed (UltraScale+ only).
    reason : str
        Human-readable explanation.
    """

    strategy: str
    neuron_count: int
    total_bits: int
    bram_18k_used: int = 0
    bram_36k_used: int = 0
    uram_used: int = 0
    reason: str = ""


def storage_recommendation(
    neuron_count: int,
    state_bits_per_neuron: int,
    *,
    has_uram: bool = False,
    register_threshold: int = 64,
    uram_threshold: int = 16384,
) -> StorageRecommendation:
    """Determine optimal storage strategy for a neuron array.

    Decides between registers (small), BRAM (medium), and URAM (large)
    based on total state bits and target capabilities.

    Parameters
    ----------
    neuron_count : int
        Number of neurons in the array.
    state_bits_per_neuron : int
        State bits per neuron.
    has_uram : bool
        True if the target has UltraRAM.
    register_threshold : int
        Max neurons for register-based storage.
    uram_threshold : int
        Min neurons for URAM migration.

    Returns
    -------
    StorageRecommendation
        Optimal storage strategy with resource estimates.
    """
    total_bits = neuron_count * state_bits_per_neuron

    if neuron_count <= register_threshold:
        return StorageRecommendation(
            strategy="registers",
            neuron_count=neuron_count,
            total_bits=total_bits,
            reason=f"{neuron_count} neurons × {state_bits_per_neuron}b = "
            f"{total_bits}b — fits in registers.",
        )

    if has_uram and neuron_count > uram_threshold:
        # URAM: 288Kb (288 × 1024 = 294912 bits) per tile, 72b wide
        uram_tiles = max(1, (total_bits + 294911) // 294912)
        return StorageRecommendation(
            strategy="uram",
            neuron_count=neuron_count,
            total_bits=total_bits,
            uram_used=uram_tiles,
            reason=f"{neuron_count} neurons × {state_bits_per_neuron}b = "
            f"{total_bits // 1024}Kb — using {uram_tiles} URAM tiles.",
        )

    # BRAM: 18Kb or 36Kb tiles
    if total_bits <= 18 * 1024:
        bram_18k = 1
        bram_36k = 0
    else:
        bram_36k = max(1, (total_bits + 36863) // 36864)
        bram_18k = 0

    return StorageRecommendation(
        strategy="bram",
        neuron_count=neuron_count,
        total_bits=total_bits,
        bram_18k_used=bram_18k,
        bram_36k_used=bram_36k,
        reason=f"{neuron_count} neurons × {state_bits_per_neuron}b = "
        f"{total_bits // 1024}Kb — using BRAM.",
    )

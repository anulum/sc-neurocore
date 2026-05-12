# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SCPN L9 holographic memory production contracts

"""Production contracts for L9 holographic boundary-cue recovery."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from sc_neurocore.scpn.layers.l9_memory import L9_MemoryLayer, L9_StochasticParameters


class _ZeroRng:
    def choice(self, choices: list[int], size: int) -> np.ndarray:
        return np.full(size, choices[0], dtype=np.float64)

    def random(self, size: int | tuple[int, ...]) -> np.ndarray:
        return np.zeros(size, dtype=np.float64)


def test_l9_boundary_cue_generates_syndrome_and_recovery() -> None:
    layer = L9_MemoryLayer(
        L9_StochasticParameters(
            n_memory_slots=4,
            bitstream_length=16,
            retrieval_gain=0.0,
            imprint_rate=0.0,
            decay_rate=0.0,
            boundary_cue_coupling=1.0,
            rng_seed=9,
        )
    )
    layer.state = -np.ones(4, dtype=np.float64)
    vars(layer)["_rng"] = _ZeroRng()

    cue = np.array([1.0, -1.0, 1.0, -1.0], dtype=np.float64)
    out = layer.step(
        0.001,
        boundary_cue=cue,
        ebs_context={"ebs_id": "run-9", "terminal_set": ("T1", "T5"), "cgp_hash": "abc123"},
    )

    np.testing.assert_array_equal(out["state"], cue)
    np.testing.assert_array_equal(out["qec_syndrome"], np.array([1, 0, 1, 0], dtype=np.uint8))
    np.testing.assert_array_equal(out["recovery_operator"], np.array([1.0, 0.0, 1.0, 0.0]))
    assert out["boundary_context_id"] == "run-9"
    assert out["boundary_terminals"] == ("T1", "T5")
    assert 0.0 <= out["holographic_entropy"] <= 1.0
    assert out["memory_free_energy"] >= 0.0


def test_l9_rejects_invalid_boundary_cue_and_context() -> None:
    layer = L9_MemoryLayer(L9_StochasticParameters(n_memory_slots=3, bitstream_length=8, rng_seed=1))

    invalid_cues: list[Any] = [
        np.array([1.0, -1.0]),
        np.array([1.0, np.nan, -1.0]),
        np.array([1.0, 0.0, 2.0]),
    ]
    for cue in invalid_cues:
        with pytest.raises(ValueError, match="boundary_cue"):
            layer.step(0.001, boundary_cue=cue)

    with pytest.raises(ValueError, match="ebs_context"):
        layer.step(0.001, ebs_context={"terminal_set": ("T1",)})

    with pytest.raises(ValueError, match="terminal_set"):
        layer.step(0.001, ebs_context={"ebs_id": "run", "terminal_set": ("T8",)})


def test_l9_rejects_invalid_boundary_cue_coupling() -> None:
    with pytest.raises(ValueError, match="boundary_cue_coupling"):
        L9_MemoryLayer(L9_StochasticParameters(boundary_cue_coupling=-0.1))
    with pytest.raises(ValueError, match="boundary_cue_coupling"):
        L9_MemoryLayer(L9_StochasticParameters(boundary_cue_coupling=1.1))

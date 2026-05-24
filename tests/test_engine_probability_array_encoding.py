# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for Rust engine probability-array encoding

"""Rust engine probability-array encoding contracts.

The engine encoder returns packed stochastic bitstreams for each probability.
These tests assert packing geometry, probability accuracy, deterministic seeds,
and the empty-input boundary.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("sc_neurocore_engine", reason="Rust engine not built", exc_type=ImportError)

import sc_neurocore_engine as v3


def test_probability_array_encoding_packs_each_probability_row() -> None:
    probabilities = np.array([0.3, 0.5, 0.8])

    packed = v3.batch_encode(probabilities, length=1024, seed=0xACE1)

    words_per_probability = (1024 + 63) // 64
    assert len(packed) == len(probabilities)
    assert all(len(row) == words_per_probability for row in packed)


def test_probability_array_encoding_tracks_requested_rates() -> None:
    probabilities = np.array([0.25, 0.5, 0.75])

    packed = v3.batch_encode(probabilities, length=10_000, seed=42)

    for row, probability in zip(packed, probabilities, strict=True):
        encoded_rate = sum(int(word).bit_count() for word in row) / 10_000
        assert encoded_rate == pytest.approx(float(probability), abs=0.05)


def test_probability_array_encoding_is_seed_deterministic() -> None:
    probabilities = np.array([0.5, 0.5])

    first = v3.batch_encode(probabilities, length=1024, seed=42)
    second = v3.batch_encode(probabilities, length=1024, seed=42)

    assert first == second


def test_probability_array_encoding_accepts_empty_input() -> None:
    packed = v3.batch_encode(np.array([], dtype=np.float64), length=1024, seed=42)

    assert len(packed) == 0

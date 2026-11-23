# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestBipolarValidationPaths from former test_bipolar_sc.py

"""Focused suite: TestBipolarValidationPaths from former test_bipolar_sc.py."""

from __future__ import annotations

import numpy as np
import pytest
from sc_neurocore.core.bipolar import (
    bipolar_decode,
    bipolar_encode,
    bipolar_mac,
)


class TestBipolarValidationPaths:
    """Guard clauses and default-argument paths across the bipolar primitives."""

    def test_encode_rejects_non_finite_value(self) -> None:
        """A non-finite value is rejected."""
        with pytest.raises(ValueError, match="NaN or Inf"):
            bipolar_encode(float("nan"), 100, rng=np.random.default_rng(42))

    def test_encode_builds_default_rng_when_omitted(self) -> None:
        """Without a generator the encoder makes its own."""
        # rng=None exercises the internal default_rng construction; a long stream
        # still decodes near the encoded value without a caller-supplied generator.
        bits = bipolar_encode(0.5, 20000)
        assert set(np.unique(bits)).issubset({0, 1})
        assert abs(bipolar_decode(bits) - 0.5) < 0.1

    def test_decode_rejects_non_binary_bits(self) -> None:
        """Bits other than 0 and 1 are rejected."""
        with pytest.raises(ValueError, match="only 0/1 bits"):
            bipolar_decode(np.array([0, 2, 1], dtype=np.uint8))

    def test_mac_rejects_non_finite_inputs(self) -> None:
        """Non-finite inputs are rejected."""
        with pytest.raises(ValueError, match="NaN or Inf"):
            bipolar_mac(np.array([0.5, np.nan]), np.array([[0.2, 0.1]]), L=100, seed=1)

    def test_mac_rejects_multidimensional_inputs(self) -> None:
        """Multidimensional inputs are rejected."""
        with pytest.raises(ValueError, match=r"inputs must have shape \(N,\)"):
            bipolar_mac(np.array([[0.5], [0.2]]), np.array([[0.2, 0.1]]), L=100, seed=1)

    def test_mac_rejects_non_2d_weights(self) -> None:
        """Weights that are not 2-D are rejected."""
        with pytest.raises(ValueError, match=r"weights must have shape \(M, N\)"):
            bipolar_mac(np.array([0.5, 0.2]), np.array([0.2, 0.1]), L=100, seed=1)

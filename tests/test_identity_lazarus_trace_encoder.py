# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestTraceEncoder from former test_identity_lazarus.py

"""Focused suite: TestTraceEncoder from former test_identity_lazarus.py."""

from __future__ import annotations

import numpy as np
from sc_neurocore.identity.encoder import (
    TraceEncoder,
)


class TestTraceEncoder:
    """Encoding text as spike traces."""

    def test_encode_shape(self) -> None:
        """A text encodes to the asked raster shape."""
        enc = TraceEncoder(n_neurons=50, hash_dims=32, seed=42)
        pattern = enc.encode("test input text", duration_ms=100, dt=0.001)
        assert pattern.shape == (50, 100)

    def test_encode_binary(self) -> None:
        """Encoded rasters are binary."""
        enc = TraceEncoder(n_neurons=50, seed=42)
        pattern = enc.encode("hello world", duration_ms=50, dt=0.001)
        assert set(np.unique(pattern)).issubset({0, 1})

    def test_different_texts_different_patterns(self) -> None:
        """Different texts give different patterns."""
        enc = TraceEncoder(n_neurons=100, seed=42)
        p1 = enc.encode("alpha beta gamma", duration_ms=100, dt=0.001)
        p2 = enc.encode("completely different content", duration_ms=100, dt=0.001)
        assert not np.array_equal(p1, p2)

    def test_deterministic(self) -> None:
        """The same text gives the same pattern."""
        enc1 = TraceEncoder(n_neurons=50, seed=42)
        enc2 = TraceEncoder(n_neurons=50, seed=42)
        p1 = enc1.encode("same text", duration_ms=50, dt=0.001)
        p2 = enc2.encode("same text", duration_ms=50, dt=0.001)
        np.testing.assert_array_equal(p1, p2)

    def test_empty_text(self) -> None:
        """An empty text still gives a raster."""
        enc = TraceEncoder(n_neurons=50, seed=42)
        pattern = enc.encode("", duration_ms=50, dt=0.001)
        assert pattern.shape == (50, 50)

    def test_encode_key_value(self) -> None:
        """A key-value pair encodes to a raster."""
        enc = TraceEncoder(n_neurons=100, seed=42)
        pattern = enc.encode_key_value("project", "sc-neurocore")
        assert isinstance(pattern, np.ndarray)
        assert pattern.shape[0] == 100

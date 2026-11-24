# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestQuantizeDelays from former test_compression_toolkit.py

"""Focused suite: TestQuantizeDelays from former test_compression_toolkit.py."""

from __future__ import annotations

import numpy as np
from sc_neurocore.compression import (
    quantize_delays,
)


class TestQuantizeDelays:
    """Quantising synaptic delays."""

    def test_basic(self) -> None:
        """Delays round to whole steps."""
        d = np.array([0.5, 1.7, 3.2, 5.0])
        q = quantize_delays(d, resolution=1)
        np.testing.assert_array_equal(q, np.array([0, 2, 3, 5]))

    def test_resolution_2(self) -> None:
        """A resolution of 2 gives even delays."""
        d = np.array([1.0, 2.5, 3.8, 7.0])
        q = quantize_delays(d, resolution=2)
        assert np.all(q % 2 == 0)

    def test_max_delay(self) -> None:
        """Delays are capped at the maximum."""
        d = np.array([1.0, 5.0, 10.0, 100.0])
        q = quantize_delays(d, resolution=1, max_delay=8)
        assert q.max() <= 8

    def test_negative_clamped(self) -> None:
        """Negative delays are clamped to 0."""
        d = np.array([-1.0, 0.0, 1.0])
        q = quantize_delays(d, resolution=1)
        assert q[0] == 0

    def test_dtype(self) -> None:
        """Quantised delays are int64."""
        d = np.array([1.5, 2.5])
        q = quantize_delays(d, resolution=1)
        assert q.dtype == np.int64

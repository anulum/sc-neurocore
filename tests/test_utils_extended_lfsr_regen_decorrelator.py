# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestLFSRRegenDecorrelator from former test_utils_extended.py

"""Focused suite: TestLFSRRegenDecorrelator from former test_utils_extended.py."""

from __future__ import annotations

import numpy as np
import pytest
from sc_neurocore.utils.decorrelators import (
    LFSRRegenDecorrelator,
)


class TestLFSRRegenDecorrelator:
    """Decorrelating bitstreams by LFSR regeneration."""

    def test_preserves_approximate_probability(self) -> None:
        """Regenerated stream should have approximately the same probability."""
        rng = np.random.default_rng(0)
        bs = (rng.random(2048) < 0.4).astype(np.uint8)
        dec = LFSRRegenDecorrelator(seed=42)
        result = dec.process(bs)
        assert result.mean() == pytest.approx(bs.mean(), abs=0.05)
        assert len(result) == len(bs)

    def test_output_is_binary(self) -> None:
        """The decorrelated stream holds only zeros and ones."""
        bs = np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=np.uint8)
        dec = LFSRRegenDecorrelator(seed=0)
        result = dec.process(bs)
        assert set(np.unique(result)).issubset({0, 1})

    def test_produces_different_sequence(self) -> None:
        """Regenerated bitstream should not be identical to input."""
        rng = np.random.default_rng(0)
        bs = (rng.random(512) < 0.5).astype(np.uint8)
        dec = LFSRRegenDecorrelator(seed=77)
        result = dec.process(bs)
        # With different RNG seed, almost certainly different
        assert not np.array_equal(result, bs)

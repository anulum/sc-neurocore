# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestTemporalAccumulatedBN from former test_spike_norm.py

"""Focused suite: TestTemporalAccumulatedBN from former test_spike_norm.py."""

from __future__ import annotations

import numpy as np
from sc_neurocore.spike_norm import (
    TemporalAccumulatedBN,
)


class TestTemporalAccumulatedBN:
    """Temporally accumulated batch normalisation and its reset."""

    def test_forward(self) -> None:
        """The output has the expected shape."""
        bn = TemporalAccumulatedBN(n_features=8)
        for _ in range(5):
            out = bn.forward(np.random.randn(4, 8), training=True)
        assert out.shape == (4, 8)

    def test_reset(self) -> None:
        """A reset after drive leaves a valid state."""
        bn = TemporalAccumulatedBN(n_features=4)
        bn.forward(np.ones(4), training=True)
        bn.reset()
        assert np.allclose(bn._accumulated, 0)

# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSTGDynamics from former test_model_marder_stg.py

"""Focused suite: TestSTGDynamics from former test_model_marder_stg.py."""

from __future__ import annotations

import numpy as np
import pytest
from sc_neurocore.neurons.models.marder_stg import (
    MarderSTGNeuron,
)
from tests.model_marder_stg_support import (
    _run,
)


class TestSTGDynamics:
    """Dynamics of the MarderSTGNeuron."""

    def test_rate_increases_with_drive(self) -> None:
        """Rate increases with drive."""
        low = len(_run(MarderSTGNeuron(), current=0.0, steps=50_000))
        high = len(_run(MarderSTGNeuron(), current=10.0, steps=50_000))
        assert high > low

    @pytest.mark.parametrize("current", [0.0, 2.0, 5.0, 10.0])
    def test_fi_sweep_finite(self, current: float) -> None:
        """The voltage stays finite across a current sweep."""
        n = MarderSTGNeuron()
        for _ in range(20_000):
            n.step(current)
        assert np.isfinite(n.v)

    def test_upward_crossing_only(self) -> None:
        """Upward crossing only."""
        n = MarderSTGNeuron()
        prev_v = n.v
        for _ in range(50_000):
            if n.step(0.0) == 1:
                assert prev_v < n.v_threshold
            prev_v = n.v

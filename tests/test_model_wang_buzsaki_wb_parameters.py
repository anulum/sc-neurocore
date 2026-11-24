# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestWBParameters from former test_model_wang_buzsaki.py

"""Focused suite: TestWBParameters from former test_model_wang_buzsaki.py."""

from __future__ import annotations

from typing import Any
import numpy as np
import pytest
from sc_neurocore.neurons.models.wang_buzsaki import (
    WangBuzsakiNeuron,
)


class TestWBParameters:
    """Parameter behaviour of the WangBuzsakiNeuron."""

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"v": np.nan}, "v"),
            ({"h": np.inf}, "h"),
            ({"n": np.nan}, "n"),
            ({"g_na": 0.0}, "g_na"),
            ({"g_k": -1.0}, "g_k"),
            ({"g_l": np.nan}, "g_l"),
            ({"c_m": 0.0}, "c_m"),
            ({"phi": 0.0}, "phi"),
            ({"dt": 0.0}, "dt"),
        ],
    )
    def test_rejects_invalid_numerical_configuration(
        self, kwargs: dict[str, Any], match: str
    ) -> None:
        """An invalid numerical configuration is rejected."""
        with pytest.raises(ValueError, match=match):
            WangBuzsakiNeuron(**kwargs)

    def test_rejects_non_finite_current_before_state_mutation(self) -> None:
        """A non-finite current is rejected before any state change."""
        n = WangBuzsakiNeuron()
        state = (n.v, n.h, n.n)
        with pytest.raises(ValueError, match="current"):
            n.step(np.nan)
        assert (n.v, n.h, n.n) == state

    def test_rejects_corrupted_runtime_state_before_mutation(self) -> None:
        """Corrupted runtime state is rejected before any change."""
        n = WangBuzsakiNeuron()
        n.h = np.inf
        state = (n.v, n.h, n.n)
        with pytest.raises(FloatingPointError, match="state"):
            n.step(1.0)
        assert (n.v, n.h, n.n) == state

    def test_rejects_rate_overflow_before_state_mutation(self) -> None:
        """The model rejects rate overflow before state mutation."""
        n = WangBuzsakiNeuron(v=-1.0e308)
        state = (n.v, n.h, n.n)
        with pytest.raises(FloatingPointError, match="rate overflowed"):
            n.step(1.0)
        assert (n.v, n.h, n.n) == state

    @pytest.mark.parametrize("dt", [0.005, 0.01, 0.02])
    def test_dt_stability(self, dt: float) -> None:
        """The state stays finite across time steps."""
        n = WangBuzsakiNeuron(dt=dt)
        for _ in range(10000):
            n.step(2.0)
        assert np.isfinite(n.v)

    def test_deterministic(self) -> None:
        """The same inputs give the same trace."""
        traces = []
        for _ in range(2):
            n = WangBuzsakiNeuron()
            trace = [(n.step(2.0), n.v) for _ in range(100)]
            traces.append(trace)
        assert traces[0] == traces[1]

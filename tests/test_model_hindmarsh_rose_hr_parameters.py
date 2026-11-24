# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestHRParameters from former test_model_hindmarsh_rose.py

"""Focused suite: TestHRParameters from former test_model_hindmarsh_rose.py."""

from __future__ import annotations

from typing import cast
from typing import Any
import numpy as np
import pytest
from sc_neurocore.neurons.models.hindmarsh_rose import (
    HindmarshRoseNeuron,
)


class TestHRParameters:
    """Parameter behaviour of the HindmarshRoseNeuron."""

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("dt", 0.0),
            ("dt", float("nan")),
            ("r", -0.001),
            ("s", 0.0),
            ("b", float("inf")),
        ],
    )
    def test_rejects_nonphysical_parameters(self, field: str, value: float) -> None:
        """The model rejects nonphysical parameters."""
        with pytest.raises(ValueError, match=field):
            kwargs: dict[str, Any] = {field: value}
            HindmarshRoseNeuron(**kwargs)

    def test_rejects_unknown_integrator(self) -> None:
        """The model rejects unknown integrator."""
        with pytest.raises(ValueError, match="integrator"):
            HindmarshRoseNeuron(integrator=cast(Any, "verlet"))

    @pytest.mark.parametrize("b", [2.0, 3.0, 4.0])
    def test_b_sweep(self, b: float) -> None:
        """The model is exercised across a sweep of b."""
        n = HindmarshRoseNeuron(b=b)
        for _ in range(10_000):
            n.step(5.0)
        assert np.isfinite(n.x)

    @pytest.mark.parametrize("r", [0.0005, 0.001, 0.005])
    def test_r_slow_timescale(self, r: float) -> None:
        """The slow variable stays finite across r values."""
        n = HindmarshRoseNeuron(r=r)
        for _ in range(10_000):
            n.step(5.0)
        assert np.isfinite(n.z)

    @pytest.mark.parametrize("dt", [0.05, 0.1, 0.15])
    def test_dt_stability(self, dt: float) -> None:
        """The state stays finite across time steps."""
        n = HindmarshRoseNeuron(dt=dt)
        for _ in range(10_000):
            n.step(5.0)
        assert np.isfinite(n.x) and np.isfinite(n.y) and np.isfinite(n.z)

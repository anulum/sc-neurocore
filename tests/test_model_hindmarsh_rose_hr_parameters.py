# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestHRParameters from former test_model_hindmarsh_rose.py

"""Focused suite: TestHRParameters from former test_model_hindmarsh_rose.py."""

from __future__ import annotations

from tests.model_hindmarsh_rose_support import *  # noqa: F403


class TestHRParameters:
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
    def test_rejects_nonphysical_parameters(self, field: str, value: float):
        with pytest.raises(ValueError, match=field):
            HindmarshRoseNeuron(**{field: value})

    def test_rejects_unknown_integrator(self):
        with pytest.raises(ValueError, match="integrator"):
            HindmarshRoseNeuron(integrator="verlet")

    @pytest.mark.parametrize("b", [2.0, 3.0, 4.0])
    def test_b_sweep(self, b: float):
        n = HindmarshRoseNeuron(b=b)
        for _ in range(10_000):
            n.step(5.0)
        assert np.isfinite(n.x)

    @pytest.mark.parametrize("r", [0.0005, 0.001, 0.005])
    def test_r_slow_timescale(self, r: float):
        n = HindmarshRoseNeuron(r=r)
        for _ in range(10_000):
            n.step(5.0)
        assert np.isfinite(n.z)

    @pytest.mark.parametrize("dt", [0.05, 0.1, 0.15])
    def test_dt_stability(self, dt: float):
        n = HindmarshRoseNeuron(dt=dt)
        for _ in range(10_000):
            n.step(5.0)
        assert np.isfinite(n.x) and np.isfinite(n.y) and np.isfinite(n.z)

# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestWendlingParameters from former test_model_wendling.py

"""Focused suite: TestWendlingParameters from former test_model_wendling.py."""

from __future__ import annotations

from tests.model_wendling_support import *  # noqa: F403

class TestWendlingParameters:
    @pytest.mark.parametrize("dt", [0.0005, 0.001, 0.002])
    def test_dt_stability(self, dt: float):
        n = WendlingNeuron(dt=dt)
        for _ in range(50000):
            n.step(220.0)
        assert np.isfinite(n.y1)

    def test_deterministic(self):
        traces = []
        for _ in range(2):
            n = WendlingNeuron()
            trace = [n.step(220.0) for _ in range(200)]
            traces.append(trace)
        assert traces[0] == traces[1]

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"dt": 0.0},
            {"a_exc": 0.0},
            {"b_fast": 0.0},
            {"g_slow": 0.0},
            {"a_rate": 0.0},
            {"b_rate": 0.0},
            {"g_rate": 0.0},
            {"c": -1.0},
            {"e0": 0.0},
            {"r": 0.0},
            {"y0": math.nan},
            {"v0": math.inf},
        ],
    )
    def test_invalid_physical_configuration_is_rejected(self, kwargs: dict[str, float]):
        with pytest.raises(ValueError):
            WendlingNeuron(**kwargs)

    def test_non_finite_external_input_does_not_mutate_state(self):
        n = WendlingNeuron()
        before = (
            n.y0,
            n.y5,
            n.y1,
            n.y6,
            n.y2,
            n.y7,
            n.y3,
            n.y8,
            n.y4,
            n.y9,
        )

        with pytest.raises(ValueError):
            n.step(math.nan)

        assert (
            n.y0,
            n.y5,
            n.y1,
            n.y6,
            n.y2,
            n.y7,
            n.y3,
            n.y8,
            n.y4,
            n.y9,
        ) == before

    def test_corrupted_runtime_state_does_not_mutate_state(self):
        n = WendlingNeuron()
        n.y6 = math.inf
        before = (
            n.y0,
            n.y5,
            n.y1,
            n.y6,
            n.y2,
            n.y7,
            n.y3,
            n.y8,
            n.y4,
            n.y9,
        )

        with pytest.raises(ValueError):
            n.step(220.0)

        assert (
            n.y0,
            n.y5,
            n.y1,
            n.y6,
            n.y2,
            n.y7,
            n.y3,
            n.y8,
            n.y4,
            n.y9,
        ) == before

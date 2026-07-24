# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestTransform from former test_gpfa.py

"""Focused suite: TestTransform from former test_gpfa.py."""

from __future__ import annotations

from tests.gpfa_support import *  # noqa: F403


class TestTransform:
    def test_projects_new_trains(self) -> None:
        trains = _synthetic_trains()
        params = gpfa(trains, n_latents=3, bin_ms=20.0, max_iter=20)
        proj = gpfa_transform(trains, params, bin_ms=20.0)
        assert proj.shape == params["trajectories"].shape

    def test_empty_inputs_return_empty(self) -> None:
        full = {"C": np.zeros((1, 1)), "d": np.zeros(1), "R": np.eye(1), "tau": np.ones(1)}
        assert gpfa_transform([], full).size == 0  # no trains
        empty_c = {"C": np.array([]), "d": np.array([]), "R": np.array([]), "tau": np.array([])}
        assert gpfa_transform(_synthetic_trains(3, 120), empty_c).size == 0  # untrained params

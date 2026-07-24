# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestGpfa from former test_gpfa.py

"""Focused suite: TestGpfa from former test_gpfa.py."""

from __future__ import annotations

from tests.gpfa_support import *  # noqa: F403


class TestGpfa:
    def test_deterministic_and_seed_independent(self) -> None:
        trains = _synthetic_trains()
        a = gpfa(trains, n_latents=3, bin_ms=20.0, max_iter=30, seed=1)
        b = gpfa(trains, n_latents=3, bin_ms=20.0, max_iter=30, seed=999)
        npt.assert_array_equal(a["trajectories"], b["trajectories"])
        npt.assert_array_equal(a["C"], b["C"])

    def test_auto_matches_python_within_tolerance(self) -> None:
        # `auto` selects the fastest available backend (Rust when the engine is
        # present, else NumPy); either way it agrees with the NumPy reference up to
        # floating-point round-off.
        trains = _synthetic_trains()
        auto = gpfa(trains, n_latents=2, bin_ms=20.0, max_iter=20, backend="auto")
        py = gpfa(trains, n_latents=2, bin_ms=20.0, max_iter=20, backend="python")
        npt.assert_allclose(auto["trajectories"], py["trajectories"], atol=1e-7)

    def test_clamps_latent_count(self) -> None:
        trains = _synthetic_trains(n_neurons=2, n_samples=120)
        result = gpfa(trains, n_latents=9, bin_ms=20.0, max_iter=5)
        assert result["C"].shape[1] <= 2

    def test_empty_input_returns_empty(self) -> None:
        result = gpfa([], n_latents=3)
        assert result["trajectories"].size == 0
        assert result["log_likelihoods"] == []

    def test_unknown_backend_rejected(self) -> None:
        trains = _synthetic_trains(n_neurons=3, n_samples=120)
        with pytest.raises(ValueError, match="not available"):
            gpfa(trains, n_latents=2, bin_ms=20.0, max_iter=3, backend="cuda")

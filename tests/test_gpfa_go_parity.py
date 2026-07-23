# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestGoParity from former test_gpfa.py

"""Focused suite: TestGoParity from former test_gpfa.py."""

from __future__ import annotations

from tests.gpfa_support import *  # noqa: F403

@pytest.mark.skipif(not _GO_AVAILABLE, reason="Go GPFA library not built")
class TestGoParity:
    """The Go backend matches the NumPy reference up to float64 round-off."""

    def test_full_pipeline_parity(self) -> None:
        trains = _synthetic_trains(6, 400)
        py = gpfa(trains, n_latents=2, bin_ms=20.0, max_iter=30, backend="python")
        go = gpfa(trains, n_latents=2, bin_ms=20.0, max_iter=30, backend="go")
        assert len(py["log_likelihoods"]) == len(go["log_likelihoods"])
        npt.assert_allclose(go["trajectories"], py["trajectories"], atol=1e-8)
        npt.assert_allclose(go["C"], py["C"], atol=1e-8)
        npt.assert_allclose(go["R"], py["R"], atol=1e-9)
        npt.assert_allclose(go["log_likelihoods"], py["log_likelihoods"], atol=1e-6)

    def test_ensure_go_is_cached(self) -> None:
        assert _GPFA_MODULE._ensure_go_gpfa() is True
        assert _GPFA_MODULE._ensure_go_gpfa() is True

# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestMojoParity from former test_gpfa.py

"""Focused suite: TestMojoParity from former test_gpfa.py."""

from __future__ import annotations

from tests.gpfa_support import *  # noqa: F403


@pytest.mark.skipif(not _MOJO_AVAILABLE, reason="Mojo GPFA library not built")
class TestMojoParity:
    """The Mojo backend matches the NumPy reference up to float64 round-off."""

    def test_full_pipeline_parity(self) -> None:
        trains = _synthetic_trains(6, 400)
        py = gpfa(trains, n_latents=2, bin_ms=20.0, max_iter=30, backend="python")
        mo = gpfa(trains, n_latents=2, bin_ms=20.0, max_iter=30, backend="mojo")
        assert len(py["log_likelihoods"]) == len(mo["log_likelihoods"])
        npt.assert_allclose(mo["trajectories"], py["trajectories"], atol=1e-8)
        npt.assert_allclose(mo["C"], py["C"], atol=1e-8)
        npt.assert_allclose(mo["R"], py["R"], atol=1e-9)
        npt.assert_allclose(mo["log_likelihoods"], py["log_likelihoods"], atol=1e-6)

    def test_ensure_mojo_is_cached(self) -> None:
        assert _GPFA_MODULE._ensure_mojo_gpfa() is True
        assert _GPFA_MODULE._ensure_mojo_gpfa() is True

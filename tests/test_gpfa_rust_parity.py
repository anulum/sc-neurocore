# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestRustParity from former test_gpfa.py

"""Focused suite: TestRustParity from former test_gpfa.py."""

from __future__ import annotations

from tests.gpfa_support import *  # noqa: F403

@pytest.mark.skipif(not _RUST_AVAILABLE, reason="Rust GPFA backend not built")
class TestRustParity:
    """The Rust backend matches the NumPy reference up to float64 round-off."""

    def test_full_pipeline_parity(self) -> None:
        trains = _synthetic_trains()
        py = gpfa(trains, n_latents=3, bin_ms=20.0, max_iter=40, backend="python")
        ru = gpfa(trains, n_latents=3, bin_ms=20.0, max_iter=40, backend="rust")
        assert len(py["log_likelihoods"]) == len(ru["log_likelihoods"])
        npt.assert_allclose(ru["trajectories"], py["trajectories"], atol=1e-7)
        npt.assert_allclose(ru["C"], py["C"], atol=1e-7)
        npt.assert_allclose(ru["d"], py["d"], atol=1e-9)
        npt.assert_allclose(ru["R"], py["R"], atol=1e-9)
        npt.assert_allclose(ru["log_likelihoods"], py["log_likelihoods"], atol=1e-6)

    def test_auto_selects_rust(self) -> None:
        # The structured nalgebra Rust path is the fastest measured backend, so
        # `auto` resolves to it when the engine is present (identical result).
        trains = _synthetic_trains()
        auto = gpfa(trains, n_latents=2, bin_ms=20.0, max_iter=20, backend="auto")
        rust = gpfa(trains, n_latents=2, bin_ms=20.0, max_iter=20, backend="rust")
        npt.assert_array_equal(auto["trajectories"], rust["trajectories"])

# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestJuliaParity from former test_phi_estimation.py

"""Focused suite: TestJuliaParity from former test_phi_estimation.py."""

from __future__ import annotations

from tests.phi_estimation_support import *  # noqa: F403

@pytest.mark.skipif(not _JULIA_AVAILABLE, reason="juliacall not installed")
class TestJuliaParity:
    def test_parity(self) -> None:
        rng = np.random.RandomState(13)
        data = np.vstack([rng.randn(200) for _ in range(4)])
        py = phi_star(data, tau=1, backend="python")
        ju = phi_star(data, tau=1, backend="julia")
        npt.assert_allclose(ju, py, atol=1e-9)

    def test_ensure_julia_is_cached(self) -> None:
        assert _PHI_MODULE._ensure_julia_phi() is True
        assert _PHI_MODULE._ensure_julia_phi() is True

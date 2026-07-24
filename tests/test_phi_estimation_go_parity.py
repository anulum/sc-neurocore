# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestGoParity from former test_phi_estimation.py

"""Focused suite: TestGoParity from former test_phi_estimation.py."""

from __future__ import annotations

from tests.phi_estimation_support import *  # noqa: F403


@pytest.mark.skipif(not _GO_AVAILABLE, reason="Go Phi library not built")
class TestGoParity:
    def test_parity(self) -> None:
        rng = np.random.RandomState(17)
        for n in (2, 4, 6):
            data = np.vstack([rng.randn(200) for _ in range(n)])
            py = phi_star(data, tau=1, backend="python")
            go = phi_star(data, tau=1, backend="go")
            npt.assert_allclose(go, py, atol=1e-9)

    def test_ensure_go_is_cached(self) -> None:
        assert _PHI_MODULE._ensure_go_phi() is True
        assert _PHI_MODULE._ensure_go_phi() is True

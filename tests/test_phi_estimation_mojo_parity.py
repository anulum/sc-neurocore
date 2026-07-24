# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestMojoParity from former test_phi_estimation.py

"""Focused suite: TestMojoParity from former test_phi_estimation.py."""

from __future__ import annotations

from tests.phi_estimation_support import *  # noqa: F403


@pytest.mark.skipif(not _MOJO_AVAILABLE, reason="Mojo Phi library not built")
class TestMojoParity:
    def test_parity(self) -> None:
        rng = np.random.RandomState(19)
        for n in (2, 4, 6):
            data = np.vstack([rng.randn(200) for _ in range(n)])
            py = phi_star(data, tau=1, backend="python")
            mo = phi_star(data, tau=1, backend="mojo")
            npt.assert_allclose(mo, py, atol=1e-7)

    def test_ensure_mojo_is_cached(self) -> None:
        assert _PHI_MODULE._ensure_mojo_phi() is True
        assert _PHI_MODULE._ensure_mojo_phi() is True

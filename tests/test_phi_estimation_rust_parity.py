# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestRustParity from former test_phi_estimation.py

"""Focused suite: TestRustParity from former test_phi_estimation.py."""

from __future__ import annotations

from tests.phi_estimation_support import *  # noqa: F403

@pytest.mark.skipif(not _RUST_AVAILABLE, reason="Rust Phi backend not built")
class TestRustParity:
    def test_parity_across_sizes(self) -> None:
        rng = np.random.RandomState(11)
        for n in (2, 3, 5, 8):
            data = np.vstack([rng.randn(220) for _ in range(n)])
            py = phi_star(data, tau=1, backend="python")
            ru = phi_star(data, tau=1, backend="rust")
            npt.assert_allclose(ru, py, atol=1e-9)

    def test_auto_selects_rust(self) -> None:
        data = _correlated(n_channels=4)
        npt.assert_array_equal(
            phi_star(data, tau=1, backend="auto"), phi_star(data, tau=1, backend="rust")
        )

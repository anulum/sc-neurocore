# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSoftplus from former test_gen_vmin_lif_lut.py

"""Focused suite: TestSoftplus from former test_gen_vmin_lif_lut.py."""

from __future__ import annotations

from gen_vmin_lif_lut_support import *  # noqa: F403


class TestSoftplus:
    def test_softplus_zero(self) -> None:
        assert softplus_float(0.0, 1.0) == pytest.approx(math.log(2), abs=1e-9)

    def test_softplus_large_positive_linear(self) -> None:
        # softplus(z) ≈ z for large z (matches PyTorch threshold=20 behaviour)
        assert softplus_float(50.0, 1.0) == pytest.approx(50.0, abs=1e-9)

    def test_softplus_negative(self) -> None:
        # softplus(-5) ≈ log(1 + e^-5) ≈ 0.00671
        assert softplus_float(-5.0, 1.0) == pytest.approx(0.00671535, abs=1e-5)

    def test_softplus_beta_scaling(self) -> None:
        # softplus(z, beta=2) = (1/2) * log(1 + e^(2z))
        # at z=1: (1/2) * log(1 + e^2) ≈ 0.5 * log(8.389) ≈ 1.063
        assert softplus_float(1.0, 2.0) == pytest.approx(1.0634640, abs=1e-5)

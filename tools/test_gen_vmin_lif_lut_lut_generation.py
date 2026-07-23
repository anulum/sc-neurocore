# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestLUTGeneration from former test_gen_vmin_lif_lut.py

"""Focused suite: TestLUTGeneration from former test_gen_vmin_lif_lut.py."""

from __future__ import annotations

from gen_vmin_lif_lut_support import *  # noqa: F403

class TestLUTGeneration:
    def test_lut_size(self) -> None:
        lut = gen_softplus_lut(beta=1.0, size=LUT_SIZE, z_max=LUT_RANGE)
        assert len(lut) == LUT_SIZE

    def test_lut_monotonic(self) -> None:
        lut = gen_softplus_lut(beta=1.0, size=LUT_SIZE, z_max=LUT_RANGE)
        for i in range(len(lut) - 1):
            assert lut[i] <= lut[i + 1], f"non-monotonic at index {i}"

    def test_lut_first_entry_is_log2(self) -> None:
        lut = gen_softplus_lut(beta=1.0, size=LUT_SIZE, z_max=LUT_RANGE)
        # softplus(0) = log(2) ≈ 0.6931
        assert decode_q88(lut[0]) == pytest.approx(math.log(2), abs=1.0 / 256)

    def test_lut_last_entry_near_linear(self) -> None:
        # At z = 16 - step ≈ 15.75, softplus(z) ≈ z
        lut = gen_softplus_lut(beta=1.0, size=LUT_SIZE, z_max=LUT_RANGE)
        last_z = (LUT_SIZE - 1) * (LUT_RANGE / LUT_SIZE)
        assert decode_q88(lut[-1]) == pytest.approx(last_z, abs=0.01)

# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestLUTLookup from former test_gen_vmin_lif_lut.py

"""Focused suite: TestLUTLookup from former test_gen_vmin_lif_lut.py."""

from __future__ import annotations

from tools.gen_vmin_lif_lut import (
    decode_q88,
    encode_q88,
    gen_softplus_lut,
    lut_lookup,
    LUT_RANGE,
    LUT_SIZE,
    softplus_float,
)


class TestLUTLookup:
    """Look up softplus values in the Q8.8 table."""

    def test_lookup_zero(self) -> None:
        """Zero maps to the first table entry."""
        lut = gen_softplus_lut(beta=1.0, size=LUT_SIZE, z_max=LUT_RANGE)
        assert lut_lookup(lut, 0) == lut[0]

    def test_lookup_negative_returns_first(self) -> None:
        """A negative input clamps to the first table entry."""
        lut = gen_softplus_lut(beta=1.0, size=LUT_SIZE, z_max=LUT_RANGE)
        assert lut_lookup(lut, -100) == lut[0]

    def test_lookup_above_range_returns_input(self) -> None:
        """Above the table's range the lookup returns the input itself."""
        # For z >> LUT_RANGE, softplus(z) ≈ z
        lut = gen_softplus_lut(beta=1.0, size=LUT_SIZE, z_max=LUT_RANGE)
        z_q88 = encode_q88(20.0)
        result = lut_lookup(lut, z_q88)
        assert result == z_q88

    def test_lookup_at_lut_endpoints(self) -> None:
        """The range's lower end gives the first entry and its upper end the input."""
        lut = gen_softplus_lut(beta=1.0, size=LUT_SIZE, z_max=LUT_RANGE)
        # z = 0 → first entry
        assert lut_lookup(lut, 0) == lut[0]
        # z = LUT_RANGE → linear extension
        z_q88 = encode_q88(LUT_RANGE)
        assert lut_lookup(lut, z_q88) == z_q88

    def test_lookup_near_upper_bin_returns_last_lut_entry(self) -> None:
        """An input in the last bin returns the last table entry."""
        lut = [10, 20, 30, 40]

        assert lut_lookup(lut, encode_q88(15.0), size=len(lut), z_max=16.0) == 40

    def test_lookup_accuracy_vs_float(self) -> None:
        """Lookups stay within 0.05 or one percent of the float softplus."""
        # 1% relative or 0.05 absolute (whichever larger) on the LUT range
        lut = gen_softplus_lut(beta=1.0, size=LUT_SIZE, z_max=LUT_RANGE)
        for z in [0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0]:
            z_q88 = encode_q88(z)
            sp_lut = decode_q88(lut_lookup(lut, z_q88))
            sp_ref = softplus_float(z, 1.0)
            err = abs(sp_lut - sp_ref)
            assert err < max(0.05, 0.01 * sp_ref), (
                f"z={z}: lut={sp_lut:.4f}, ref={sp_ref:.4f}, err={err:.4f}"
            )

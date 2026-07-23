# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestValidateLpHp from former test_compiler_validation_units.py

"""Focused suite: TestValidateLpHp from former test_compiler_validation_units.py."""

from __future__ import annotations

from tests.compiler_validation_units_support import *  # noqa: F403

class TestValidateLpHp:
    """The low/high-precision pair must order widths, keep both fractions
    positive, and leave the LP datapath at least two bits wide."""

    def test_accepts_sensible_pair(self) -> None:
        _validate_lp_hp(4, 2, 8, 4)  # a well-ordered pair must validate silently

    def test_rejects_lp_width_not_below_hp(self) -> None:
        with pytest.raises(ValueError, match="must be strictly less than HP data_width"):
            _validate_lp_hp(8, 2, 8, 4)

    def test_rejects_lp_fraction_below_one(self) -> None:
        with pytest.raises(ValueError, match=r"LP fraction \(0\) must be >= 1"):
            _validate_lp_hp(4, 0, 8, 4)

    def test_rejects_hp_fraction_below_one(self) -> None:
        with pytest.raises(ValueError, match=r"HP fraction \(0\) must be >= 1"):
            _validate_lp_hp(4, 2, 8, 0)

    def test_rejects_lp_width_below_two(self) -> None:
        # lp_width=1 is below hp_width=8 and both fractions are positive, so the
        # final minimum-width guard is the one that fires.
        with pytest.raises(ValueError, match=r"LP data_width \(1\) must be >= 2"):
            _validate_lp_hp(1, 1, 8, 4)

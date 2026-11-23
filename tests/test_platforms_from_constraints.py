# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestFromConstraints from former test_platforms.py

"""Focused suite: TestFromConstraints from former test_platforms.py."""

from __future__ import annotations

from sc_neurocore.compiler.platforms import (
    get_profile,
)


class TestFromConstraints:
    """Building a platform profile from design constraints."""

    def test_basic(self) -> None:
        """A constrained profile is valid and can be retrieved by name."""
        from sc_neurocore.compiler.platforms import (
            HardwareProfile,
        )

        p = HardwareProfile.from_constraints(
            "test_w10_auto",
            vendor="TestVendor",
            platform_class="custom",
        )
        assert p.data_width >= 8
        assert p.fraction >= 1
        retrieved = get_profile("test_w10_auto")
        assert retrieved.vendor == "TestVendor"

    def test_low_power(self) -> None:
        """A low-power constraint selects an eight-bit data width."""
        from sc_neurocore.compiler.platforms import HardwareProfile

        p = HardwareProfile.from_constraints(
            "test_w10_lowpow",
            max_power_budget_mw=5,
        )
        assert p.data_width == 8

    def test_explicit_width(self) -> None:
        """An explicit width and fraction are kept as given."""
        from sc_neurocore.compiler.platforms import HardwareProfile

        p = HardwareProfile.from_constraints(
            "test_w10_32bit",
            data_width=32,
            fraction=16,
        )
        assert p.data_width == 32
        assert p.fraction == 16

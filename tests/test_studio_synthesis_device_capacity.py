# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio synthesis device capacity

"""Focused suite: TestDeviceCapacity from former test_studio_synthesis.py."""

from __future__ import annotations

from tests.studio_synthesis_support import (
    _DEVICE_CAPACITY,
    _TARGETS,
)


class TestDeviceCapacity:
    """The device capacity table of the synthesis targets."""

    def test_all_targets_have_capacity(self) -> None:
        """Every target has positive LUT and flip-flop capacities."""
        for target in _TARGETS:
            assert target in _DEVICE_CAPACITY
            cap = _DEVICE_CAPACITY[target]
            assert cap["luts"] > 0
            assert cap["ffs"] > 0

    def test_capacity_values_sane(self) -> None:
        """Every capacity stays within plausible bounds."""
        for target, cap in _DEVICE_CAPACITY.items():
            assert cap["luts"] <= 100_000
            assert cap["ffs"] <= 100_000
            assert cap["brams"] <= 500
            assert cap["dsps"] <= 500

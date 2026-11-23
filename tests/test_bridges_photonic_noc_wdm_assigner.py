# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestWDMAssigner from former test_bridges_photonic_noc.py

"""Focused suite: TestWDMAssigner from former test_bridges_photonic_noc.py."""

from __future__ import annotations

from sc_neurocore.bridges.photonic_noc import (
    WDMAssigner,
)


class TestWDMAssigner:
    """Assigning WDM wavelengths."""

    def test_assign_no_conflicts(self) -> None:
        """Three signals get three distinct wavelengths."""
        assigner = WDMAssigner(max_channels=8)
        channels = assigner.assign(["sig_a", "sig_b", "sig_c"])
        assert isinstance(channels, list)
        assert len(channels) == 3
        wavelengths = [ch.wavelength_nm for ch in channels]
        assert len(set(wavelengths)) == 3

    def test_channel_wavelength_positive(self) -> None:
        """Channels have positive wavelength and bandwidth."""
        assigner = WDMAssigner()
        channels = assigner.assign(["s1", "s2"])
        for ch in channels:
            assert ch.wavelength_nm > 0
            assert ch.bandwidth_nm > 0

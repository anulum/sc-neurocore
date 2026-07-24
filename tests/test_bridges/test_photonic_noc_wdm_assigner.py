# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestWDMAssigner from former test_photonic_noc.py

"""Focused suite: TestWDMAssigner from former test_photonic_noc.py."""

from __future__ import annotations

from photonic_noc_support import *  # noqa: F403


class TestWDMAssigner:
    """WDM channel assignment tests."""

    def test_assign_channels(self) -> None:
        assigner = WDMAssigner()
        channels = assigner.assign(["a", "b", "c"])
        assert len(channels) == 3

    def test_wavelength_spacing(self) -> None:
        assigner = WDMAssigner(channel_spacing_nm=0.8)
        channels = assigner.assign(["a", "b"])
        assert abs(channels[1].wavelength_nm - channels[0].wavelength_nm - 0.8) < 1e-10

    def test_signal_names(self) -> None:
        channels = WDMAssigner().assign(["x", "y"])
        assert channels[0].signal_name == "x"
        assert channels[1].signal_name == "y"

    # --- max_channels cap (closes task #47) ---

    def test_default_max_channels_is_96(self) -> None:
        a = WDMAssigner()
        assert a._max_channels == 96

    def test_assign_at_default_cap_succeeds(self) -> None:
        names = [f"sig{i}" for i in range(96)]
        channels = WDMAssigner().assign(names)
        assert len(channels) == 96

    def test_assign_above_default_cap_raises(self) -> None:
        import pytest

        names = [f"sig{i}" for i in range(97)]
        with pytest.raises(ValueError, match="max_channels"):
            WDMAssigner().assign(names)

    def test_explicit_smaller_cap_raises(self) -> None:
        import pytest

        names = ["a", "b", "c"]
        with pytest.raises(ValueError, match="max_channels"):
            WDMAssigner(max_channels=2).assign(names)

    def test_max_channels_zero_disables_cap(self) -> None:
        names = [f"sig{i}" for i in range(200)]
        channels = WDMAssigner(max_channels=0).assign(names)
        assert len(channels) == 200

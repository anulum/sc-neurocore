# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestFinalPlatforms from former test_platforms.py

"""Focused suite: TestFinalPlatforms from former test_platforms.py."""

from __future__ import annotations

import pytest
from sc_neurocore.compiler.platforms import (
    get_profile,
    list_profile_names,
)


class TestFinalPlatforms:
    """Optical, acoustic, fluidic and space-qualified profiles."""

    @pytest.mark.parametrize("name", ["ayar_teraphy", "intel_cpo"])
    def test_optical_io(self, name: str) -> None:
        """Each listed profile belongs to the optical_io class."""
        assert get_profile(name).platform_class == "optical_io"

    @pytest.mark.parametrize("name", ["mit_phononic", "caltech_mems_nn"])
    def test_acoustic(self, name: str) -> None:
        """Each listed profile belongs to the acoustic class."""
        assert get_profile(name).platform_class == "acoustic"

    @pytest.mark.parametrize(
        "name",
        [
            "stanford_microfluidic",
            "eth_fluidic_logic",
        ],
    )
    def test_fluidic(self, name: str) -> None:
        """Each listed profile belongs to the fluidic class."""
        assert get_profile(name).platform_class == "fluidic"

    @pytest.mark.parametrize(
        "name",
        [
            "bae_rad750_sq",
            "seakr_sbc",
            "vorago_va10820",
            "frontgrade_leon5",
        ],
    )
    def test_space_qualified(self, name: str) -> None:
        """Each listed profile belongs to the space_qualified class."""
        assert get_profile(name).platform_class == "space_qualified"

    def test_total_profiles(self) -> None:
        """The registry holds at least 164 profiles."""
        assert len(list_profile_names()) >= 164

    def test_total_classes(self) -> None:
        """The registry holds at least 28 platform classes."""
        classes = {get_profile(n).platform_class for n in list_profile_names()}
        assert len(classes) >= 28

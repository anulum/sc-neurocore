# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPIMCXLProfiles from former test_platforms.py

"""Focused suite: TestPIMCXLProfiles from former test_platforms.py."""

from __future__ import annotations

import pytest
from sc_neurocore.compiler.platforms import (
    get_profile,
)


class TestPIMCXLProfiles:
    """Processing-in-memory and CXL memory profiles."""

    @pytest.mark.parametrize(
        "name",
        [
            "upmem_pim",
            "samsung_hbm_pim",
            "sk_hynix_aim",
            "cxl_type3",
            "axdimm",
        ],
    )
    def test_profile_exists(self, name: str) -> None:
        """Each listed profile belongs to the in_memory class."""
        p = get_profile(name)
        assert p.platform_class == "in_memory"

# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestListBundledSchemas from former test_universal_dsl.py

"""Focused suite: TestListBundledSchemas from former test_universal_dsl.py."""

from __future__ import annotations

from tests.universal_dsl_support import *  # noqa: F403

class TestListBundledSchemas:
    """Test discovery of bundled schemas."""

    def test_lists_all_bundled(self) -> None:
        names = list_bundled_schemas()
        assert "lif" in names
        assert "fitzhugh_nagumo" in names
        assert "izhikevich" in names
        assert "hindmarsh_rose" in names
        assert "adex" in names

    def test_returns_sorted(self) -> None:
        names = list_bundled_schemas()
        assert names == sorted(names)

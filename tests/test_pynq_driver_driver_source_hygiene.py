# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestDriverSourceHygiene from former test_pynq_driver.py

"""Focused suite: TestDriverSourceHygiene from former test_pynq_driver.py."""

from __future__ import annotations

from tests.pynq_driver_support import *  # noqa: F403

class TestDriverSourceHygiene:
    """Driver source suppressions must stay narrow and documented."""

    def test_pynq_optional_import_uses_narrow_type_ignore(self):
        source = pynq_driver.__loader__.get_source(pynq_driver.__name__)

        assert source is not None
        assert "type: ignore[import-not-found]" in source
        assert "type: ignore  # noqa" not in source

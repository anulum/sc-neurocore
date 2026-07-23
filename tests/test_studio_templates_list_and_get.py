# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio templates list and get

"""Focused suite: TestListAndGet from former test_studio_templates.py."""

from __future__ import annotations

from tests.studio_templates_support import *  # noqa: F403

class TestListAndGet:
    def test_list_returns_all(self):
        result = list_templates()
        assert len(result) == 5
        assert all(isinstance(t, dict) for t in result)

    def test_get_existing(self):
        t = get_template("lif")
        assert t is not None
        assert t["name"] == "lif"

    def test_get_nonexistent(self):
        assert get_template("nonexistent") is None


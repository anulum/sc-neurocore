# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestCheckRoot from former test_datasets.py

"""Focused suite: TestCheckRoot from former test_datasets.py."""

from __future__ import annotations

from tests.datasets_support import *  # noqa: F403


class TestCheckRoot:
    def test_valid_root_returns_path(self, tmp_path):
        p = _check_root(tmp_path, "test", "http://test")
        assert p == tmp_path

    def test_missing_root_raises(self):
        with pytest.raises(FileNotFoundError, match="download from"):
            _check_root("/nonexistent_xyz_abc_123", "TestDS", "http://test")

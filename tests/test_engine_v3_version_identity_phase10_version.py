# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPhase10Version from former test_engine_v3_version_identity.py

"""Focused suite: TestPhase10Version from former test_engine_v3_version_identity.py."""

from __future__ import annotations

from tests.engine_v3_version_identity_support import *  # noqa: F403

class TestPhase10Version:
    def test_version(self) -> None:
        _assert_engine_version_matches_core()

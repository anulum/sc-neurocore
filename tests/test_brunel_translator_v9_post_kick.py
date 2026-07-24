# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestV9PostKick from former test_brunel_translator.py

"""Focused suite: TestV9PostKick from former test_brunel_translator.py."""

from __future__ import annotations

from tests.brunel_translator_support import *  # noqa: F403


class TestV9PostKick:
    """V9: Post-kick timing differs from V1."""

    def test_kick_after_step_flag(self):
        bp = BrunelParams()
        params = translate_v9_post_kick(bp)
        assert params.get("kick_after_step") is True

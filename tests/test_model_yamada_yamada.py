# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestYamada from former test_model_yamada.py

"""Focused suite: TestYamada from former test_model_yamada.py."""

from __future__ import annotations

from tests.model_yamada_support import *  # noqa: F403

class TestYamada:
    def test_moderate_drive_crosses_threshold_in_short_window(self):
        n = YamadaNeuron()
        assert sum(n.step(5.0) for _ in range(300)) > 0

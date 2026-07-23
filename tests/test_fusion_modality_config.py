# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestModalityConfig from former test_fusion.py

"""Focused suite: TestModalityConfig from former test_fusion.py."""

from __future__ import annotations

from tests.fusion_support import *  # noqa: F403

class TestModalityConfig:
    def test_fields(self):
        m = ModalityConfig(name="dvs", n_channels=128, dt_us=100.0)
        assert m.name == "dvs"
        assert m.n_channels == 128
        assert m.max_rate_hz == 1000.0

# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestModalityConfig from former test_fusion_multimodal.py

"""Focused suite: TestModalityConfig from former test_fusion_multimodal.py."""

from __future__ import annotations

from tests.fusion_multimodal_support import *  # noqa: F403

class TestModalityConfig:
    def test_defaults(self):
        m = ModalityConfig(name="dvs", n_channels=128, dt_us=1000.0)
        assert m.max_rate_hz == 1000.0

    def test_custom(self):
        m = ModalityConfig(name="audio", n_channels=64, dt_us=500.0, max_rate_hz=2000.0)
        assert m.name == "audio"
        assert m.n_channels == 64

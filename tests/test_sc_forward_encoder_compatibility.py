# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestEncoderCompatibility from former test_sc_forward.py

"""Focused suite: TestEncoderCompatibility from former test_sc_forward.py."""

from __future__ import annotations

from tests.sc_forward_support import *  # noqa: F403

class TestEncoderCompatibility:
    """NEU-SCPN.3 — BitstreamEncoder(length=, seed=) constructs without ranges."""

    def test_length_seed_only(self) -> None:
        encoder = BitstreamEncoder(length=1024, seed=123)
        assert encoder.x_min == 0.0
        assert encoder.x_max == 1.0
        ones_fraction = float(encoder.encode(0.6).mean())
        assert 0.5 <= ones_fraction <= 0.7

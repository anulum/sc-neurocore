# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestV11Q16 from former test_brunel_translator.py

"""Focused suite: TestV11Q16 from former test_brunel_translator.py."""

from __future__ import annotations

from tests.brunel_translator_support import *  # noqa: F403

class TestV11Q16:
    """V11: Q4.12 fixed-point — no overflow."""

    def test_no_overflow_standard_params(self):
        bp = BrunelParams(v_threshold=20.0, v_reset=10.0, weight_exc=5.0)
        params = translate_v11_q16(bp)
        assert params["data_width"] == 32
        assert params["fraction"] == 12
        assert params["v_threshold_q"] == 20 * 4096

    def test_overflow_raises(self):
        bp = BrunelParams(v_threshold=600000.0)  # exceeds 32-bit Q16.12
        import pytest

        with pytest.raises(OverflowError):
            translate_v11_q16(bp)

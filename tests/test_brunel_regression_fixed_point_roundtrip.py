# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestFixedPointRoundtrip from former test_brunel_regression.py

"""Focused suite: TestFixedPointRoundtrip from former test_brunel_regression.py."""

from __future__ import annotations

from tests.brunel_regression_support import *  # noqa: F403


class TestFixedPointRoundtrip:
    def test_q88_encode_decode(self):
        """Q8.8: encode then decode recovers floor quantisation."""
        bp = BrunelParams(v_threshold=20.0, v_reset=10.0)
        params = translate_v3_fixed_point(bp)
        frac = params["fraction"]
        scale = 1 << frac
        v_orig = bp.v_threshold
        v_q = params["v_threshold_q"]
        v_decoded = v_q / scale
        assert v_decoded == pytest.approx(v_orig, abs=1.0 / scale)

    def test_q16_encode_decode(self):
        """Q16.12: 12 fractional bits give 1/4096 precision."""
        bp = BrunelParams(v_threshold=20.0)
        params = translate_v11_q16(bp)
        scale = 1 << params["fraction"]
        v_decoded = params["v_threshold_q"] / scale
        assert v_decoded == pytest.approx(bp.v_threshold, abs=1.0 / scale)

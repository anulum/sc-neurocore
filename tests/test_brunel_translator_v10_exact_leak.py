# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestV10ExactLeak from former test_brunel_translator.py

"""Focused suite: TestV10ExactLeak from former test_brunel_translator.py."""

from __future__ import annotations

from tests.brunel_translator_support import *  # noqa: F403


class TestV10ExactLeak:
    """V10: Exact exponential leak."""

    def test_leak_factor_matches_exp(self):
        bp = BrunelParams(dt=0.1, tau_mem=20.0)
        params = translate_v10_exact_leak(bp)
        expected = np.exp(-0.1 / 20.0)
        assert abs(params["leak_factor"] - expected) < 1e-10

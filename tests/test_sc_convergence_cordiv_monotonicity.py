# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestCORDIVMonotonicity from former test_sc_convergence.py

"""Focused suite: TestCORDIVMonotonicity from former test_sc_convergence.py."""

from __future__ import annotations

from tests.sc_convergence_support import *  # noqa: F403


class TestCORDIVMonotonicity:
    """CORDIV output should increase with numerator probability."""

    def test_monotonic_in_numerator(self):
        L = 8192
        pd = 0.8
        y = generate_bernoulli_bitstream(pd, L, rng=RNG(99))
        prev = -1.0
        for pn in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
            x = generate_bernoulli_bitstream(pn, L, rng=RNG(42))
            z = sc_divide(x, y)
            result = np.mean(z)
            assert result > prev - 0.05, (
                f"not monotonic: pn={pn}, result={result:.3f}, prev={prev:.3f}"
            )
            prev = result

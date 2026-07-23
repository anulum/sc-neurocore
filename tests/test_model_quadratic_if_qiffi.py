# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestQIFFI from former test_model_quadratic_if.py

"""Focused suite: TestQIFFI from former test_model_quadratic_if.py."""

from __future__ import annotations

from tests.model_quadratic_if_support import *  # noqa: F403

class TestQIFFI:
    """f–I curve: f ∝ sqrt(I) for Type-I."""

    def test_monotonic_fi(self):
        rates = []
        for I in [0.5, 1.0, 2.0, 5.0]:
            n = QuadraticIFNeuron()
            rates.append(len(_run(n, current=I, steps=50000)))
        assert all(rates[i] < rates[i + 1] for i in range(len(rates) - 1))

    def test_sublinear_scaling(self):
        """QIF has sub-linear f-I: f(4I)/f(I) < 4 (not linear like LIF).

        Theoretical sqrt scaling is for continuous model; discrete reset
        introduces corrections. Verify monotonicity and sub-linearity.
        """
        n1 = QuadraticIFNeuron()
        n4 = QuadraticIFNeuron()
        s1 = len(_run(n1, current=1.0, steps=50000))
        s4 = len(_run(n4, current=4.0, steps=50000))
        ratio = s4 / s1 if s1 > 0 else 0
        assert 1.5 < ratio < 4.0, f"f(4I)/f(I) = {ratio:.2f}"

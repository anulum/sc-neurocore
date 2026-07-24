# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestQuantization from former test_digital_twin.py

"""Focused suite: TestQuantization from former test_digital_twin.py."""

from __future__ import annotations

from tests.digital_twin_support import *  # noqa: F403


class TestQuantization:
    def test_q88_quantization(self):
        model = FPGAMismatchModel(quantization_bits=16)
        x = np.array([0.123456789, -0.987654321, 0.5])
        q = model.quantize(x)
        # Q8.8 => 256 levels per integer, step = 1/256
        step = 1.0 / 256
        for val in q:
            remainder = abs(val) % step
            assert remainder < 1e-10 or abs(remainder - step) < 1e-10

    def test_quantize_preserves_shape(self):
        model = FPGAMismatchModel()
        x = np.random.randn(5, 3)
        assert model.quantize(x).shape == (5, 3)

    def test_quantize_zero(self):
        model = FPGAMismatchModel()
        assert model.quantize(np.array([0.0]))[0] == 0.0

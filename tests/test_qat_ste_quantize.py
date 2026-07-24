# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSteQuantize from former test_qat.py

"""Focused suite: TestSteQuantize from former test_qat.py."""

from __future__ import annotations

from tests.qat_support import *  # noqa: F403


class TestSteQuantize:
    def test_symmetric_roundtrip(self):
        x = np.array([0.0, 0.5, -0.5, 1.0, -1.0])
        q = _ste_quantize(x, bits=8, symmetric=True)
        assert q.shape == x.shape
        np.testing.assert_allclose(q, x, atol=0.01)

    def test_asymmetric(self):
        x = np.array([0.1, 0.5, 0.9, 1.3])
        q = _ste_quantize(x, bits=4, symmetric=False)
        assert q.shape == x.shape
        assert q.min() >= x.min() - 1e-6
        assert q.max() <= x.max() + 1e-6

    def test_quantize_reduces_unique_values(self):
        x = np.random.randn(100)
        q = _ste_quantize(x, bits=4, symmetric=True)
        assert len(np.unique(q)) <= 2**4

    def test_zero_preserved(self):
        x = np.array([0.0])
        assert _ste_quantize(x, bits=8)[0] == 0.0

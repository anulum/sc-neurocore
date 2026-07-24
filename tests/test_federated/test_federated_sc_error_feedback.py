# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestErrorFeedback from former test_federated_sc.py

"""Focused suite: TestErrorFeedback from former test_federated_sc.py."""

from __future__ import annotations

from federated_sc_support import *  # noqa: F403


class TestErrorFeedback:
    def test_initial_no_residual(self):
        ef = ErrorFeedback()
        g = np.array([1.0, 2.0, 3.0])
        acc = ef.accumulate(g)
        np.testing.assert_array_almost_equal(acc, g)

    def test_accumulates_residual(self):
        ef = ErrorFeedback()
        g1 = np.array([1.0, 2.0, 3.0])
        sparse = np.array([0.0, 2.0, 0.0])
        ef.update(g1, sparse)
        g2 = np.array([0.5, 0.5, 0.5])
        acc = ef.accumulate(g2)
        expected = g2 + (g1 - sparse)
        np.testing.assert_array_almost_equal(acc, expected)

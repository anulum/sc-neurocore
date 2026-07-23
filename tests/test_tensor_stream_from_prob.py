# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestFromProb from former test_tensor_stream.py

"""Focused suite: TestFromProb from former test_tensor_stream.py."""

from __future__ import annotations

from tests.tensor_stream_support import *  # noqa: F403

class TestFromProb:
    def test_creates_prob_domain(self):
        ts = TensorStream.from_prob(np.array([0.5]))
        assert ts.domain == "prob"

    def test_preserves_data(self):
        data = np.array([0.1, 0.5, 0.9])
        ts = TensorStream.from_prob(data)
        np.testing.assert_array_equal(ts.data, data)

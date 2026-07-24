# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestQuantumToProb from former test_tensor_stream.py

"""Focused suite: TestQuantumToProb from former test_tensor_stream.py."""

from __future__ import annotations

from tests.tensor_stream_support import *  # noqa: F403


class TestQuantumToProb:
    def test_roundtrip_exact(self):
        """prob → quantum → prob should be exact."""
        probs = np.array([0.0, 0.3, 0.5, 0.8, 1.0])
        ts = TensorStream.from_prob(probs)
        q = ts.to_quantum()
        ts_q = TensorStream(data=q, domain="quantum")
        recovered = ts_q.to_prob()
        np.testing.assert_allclose(recovered, probs, atol=1e-10)

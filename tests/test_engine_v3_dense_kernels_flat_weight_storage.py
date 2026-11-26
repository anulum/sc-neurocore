# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestFlatWeightStorage from former test_engine_v3_dense_kernels.py

"""Focused suite: TestFlatWeightStorage from former test_engine_v3_dense_kernels.py."""

from __future__ import annotations

import numpy as np
from tests.engine_requirement import require_engine

require_engine()
import sc_neurocore_engine as v3


class TestFlatWeightStorage:
    """Verify flat packed weight storage keeps API behavior unchanged."""

    def test_weight_roundtrip(self) -> None:
        """Weights round-trip through flat storage."""
        layer = v3.DenseLayer(4, 3, 256, seed=42)
        weights = np.array(
            [
                [0.1, 0.2, 0.3, 0.4],
                [0.4, 0.3, 0.2, 0.1],
                [0.5, 0.6, 0.7, 0.8],
            ],
            dtype=np.float64,
        )
        layer.set_weights(weights.tolist())
        got = np.array(layer.get_weights(), dtype=np.float64)
        np.testing.assert_allclose(got, weights)

    def test_forward_equivalence_vs_prepacked(self) -> None:
        """The fast path matches the prepacked path."""
        layer = v3.DenseLayer(8, 4, 512, seed=42)
        probs = np.array([0.2, 0.4, 0.6, 0.8, 0.1, 0.3, 0.5, 0.7], dtype=np.float64)
        seed = 31415
        packed = v3.batch_encode_numpy(probs, length=512, seed=seed)
        out_fast = np.asarray(layer.forward_fast(probs.tolist(), seed=seed), dtype=np.float64)
        out_prepacked = np.asarray(layer.forward_prepacked_numpy(packed), dtype=np.float64)
        np.testing.assert_allclose(out_fast, out_prepacked)

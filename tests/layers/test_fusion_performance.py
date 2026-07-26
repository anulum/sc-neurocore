# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SC fusion performance test

"""Opt-in small-vector fusion performance contract."""

import time

import numpy as np
import pytest

from sc_neurocore.layers.fusion import SCFusionLayer
from tests.layers.fusion_support import _perf_enabled


@pytest.mark.skipif(not _perf_enabled(), reason="Set SC_NEUROCORE_PERF=1 to enable perf checks.")
def test_fusion_perf_small() -> None:
    """Benchmark a small fusion call."""
    layer = SCFusionLayer(input_dims={"a": 64, "b": 64}, fusion_weights={"a": 1.0, "b": 1.0})
    data = {"a": np.random.random(64), "b": np.random.random(64)}
    start = time.perf_counter()
    _ = layer.forward(data)
    elapsed = time.perf_counter() - start
    assert elapsed < 1.5

# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestV20VectorizedNumpy from former test_brunel_translator.py

"""Focused suite: TestV20VectorizedNumpy from former test_brunel_translator.py."""

from __future__ import annotations

from tests.brunel_translator_support import *  # noqa: F403


class TestV20VectorizedNumpy:
    """V20: Vectorized matches basic dynamics."""

    def test_vectorized_fires(self):
        bp = BrunelParams(v_threshold=20.0, v_reset=10.0, weight_exc=5.0, external_rate_hz=200.0)
        params = translate_v20_vectorized_numpy(bp)
        n = params["n_total"]
        v = np.full(n, params["v_rest"])
        alpha = params["dt"] / params["tau_mem"]
        rng = np.random.default_rng(42)
        spike_count = 0
        for _ in range(100):
            ext = rng.poisson(200.0 * 0.1 / 1000.0, n)
            v += ext * params["weight_exc"]
            v += alpha * (params["v_rest"] - v)
            fired = v >= params["v_threshold"]
            spike_count += int(fired.sum())
            v[fired] = params["v_reset"]
        assert spike_count > 0, "Vectorized numpy must fire with strong drive"

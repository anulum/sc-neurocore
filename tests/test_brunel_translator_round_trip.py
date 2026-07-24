# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestRoundTrip from former test_brunel_translator.py

"""Focused suite: TestRoundTrip from former test_brunel_translator.py."""

from __future__ import annotations

from tests.brunel_translator_support import *  # noqa: F403


class TestRoundTrip:
    """Cross-variant consistency."""

    def test_v1_round_trip_nonzero_spikes(self):
        """Translate → simulate 1 neuron with Poisson drive → spike count > 0."""
        bp = BrunelParams(
            v_threshold=20.0,
            v_reset=10.0,
            weight_exc=5.0,
            external_rate_hz=200.0,
        )
        params = translate_v1_stochastic_lif(bp)
        n = StochasticLIFNeuron(**params["neuron_kwargs"])

        rng = np.random.default_rng(42)
        spikes = 0
        for _ in range(10000):
            # Poisson voltage kicks (delta-PSC)
            n_events = rng.poisson(bp.external_rate_hz * bp.dt / 1000.0)
            n.v += n_events * params["ext_weight"]
            spike = n.step(0.0)
            spikes += spike
        assert spikes > 0

    def test_all_variants_produce_params(self):
        """All 20 translators return non-empty dicts without errors."""
        bp = BrunelParams()
        translators = [
            translate_v1_stochastic_lif,
            translate_v2_rate_matched,
            translate_v3_fixed_point,
            translate_v4_hybrid,
            translate_v5_izhikevich,
            translate_v6_homeostatic,
            translate_v7_noisy,
            translate_v8_refractory,
            translate_v9_post_kick,
            translate_v10_exact_leak,
            translate_v11_q16,
            translate_v12_stdp,
            translate_v13_dot_product,
            translate_v14_sobol,
            translate_v15_jax,
            translate_v16_recurrent,
            translate_v17_memristive,
            translate_v18_numba,
            translate_v19_pytorch_cuda,
            translate_v20_vectorized_numpy,
        ]
        for fn in translators:
            result = fn(bp)
            assert isinstance(result, dict), f"{fn.__name__} returned non-dict"
            assert len(result) > 0, f"{fn.__name__} returned empty dict"

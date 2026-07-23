# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestV15Jax from former test_brunel_translator.py

"""Focused suite: TestV15Jax from former test_brunel_translator.py."""

from __future__ import annotations

from tests.brunel_translator_support import *  # noqa: F403

class TestV15Jax:
    """V15: JAX layer produces output."""

    def test_jax_layer_runs(self):
        try:
            from sc_neurocore import JaxSCDenseLayer
            from sc_neurocore.accel.jax_backend import jnp, HAS_JAX

            if not HAS_JAX:
                import pytest

                pytest.skip("JAX not installed")
        except (ImportError, RuntimeError):
            import pytest

            pytest.skip("JAX not installed")

        bp = BrunelParams()
        params = translate_v15_jax(bp)
        layer = JaxSCDenseLayer(
            n_neurons=10,
            n_inputs=10,
            neuron_params=params["neuron_params"],
            seed=42,
        )
        I_t = jnp.ones(10) * 5.0
        spikes = layer.step(I_t)
        assert spikes.shape == (10,)

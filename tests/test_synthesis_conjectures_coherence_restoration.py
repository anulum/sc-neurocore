# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestCoherenceRestoration from former test_synthesis_conjectures.py

"""Focused suite: TestCoherenceRestoration from former test_synthesis_conjectures.py."""

from __future__ import annotations

from tests.synthesis_conjectures_support import *  # noqa: F403


class TestCoherenceRestoration:
    """After checkpoint restore, network should recover activity.
    This tests the engineering fact, not the consciousness claim."""

    def test_checkpoint_preserves_weights(self):
        sub = IdentitySubstrate(n_cortical=30, n_inhibitory=10, n_memory=5, seed=42)
        sub.run(duration=0.05, dt=0.001)

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            path = f.name
        try:
            Checkpoint.save(sub, path)
            restored = Checkpoint.load(path)

            # Weights should be identical
            np.testing.assert_array_equal(
                sub.proj_ee.data, restored.proj_ee.data, err_msg="weights differ after checkpoint"
            )
        finally:
            os.remove(path)

    def test_restored_network_has_weights(self):
        """After restore, weight structure should be intact."""
        sub = IdentitySubstrate(n_cortical=30, n_inhibitory=10, n_memory=5, seed=42)
        sub.run(duration=0.05, dt=0.001)
        w_before = sub.proj_ee.data.copy()

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            path = f.name
        try:
            Checkpoint.save(sub, path)
            restored = Checkpoint.load(path)
            w_after = restored.proj_ee.data
            np.testing.assert_array_almost_equal(
                w_before, w_after, decimal=10, err_msg="weights changed after restore"
            )
        finally:
            os.remove(path)

    def test_population_reset_clears_state(self):
        """reset_all() should return neurons to initial conditions."""
        pop = Population(StochasticLIFNeuron, n=10, label="test")
        # Drive neuron to near-threshold
        for neuron in pop.neurons:
            neuron.step(0.5)  # inject current
        pop.reset_all()
        for neuron in pop.neurons:
            assert neuron.v == 0.0, "reset did not clear voltage"

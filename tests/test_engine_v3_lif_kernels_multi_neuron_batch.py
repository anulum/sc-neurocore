# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestMultiNeuronBatch from former test_engine_v3_lif_kernels.py

"""Focused suite: TestMultiNeuronBatch from former test_engine_v3_lif_kernels.py."""

from __future__ import annotations

from tests.engine_v3_lif_kernels_support import *  # noqa: F403

class TestMultiNeuronBatch:
    """Test parallel multi-neuron LIF batch."""

    def test_shape_and_dtype(self) -> None:
        """Output shape is (n_neurons, n_steps)."""
        currents = np.full(10, 128, dtype=np.int16)
        spikes, voltages = v3.batch_lif_run_multi(10, 100, 20, 256, currents)
        assert np.asarray(spikes).shape == (10, 100)
        assert np.asarray(voltages).shape == (10, 100)

    def test_matches_sequential(self) -> None:
        """Parallel multi-neuron must match N sequential single-neuron runs."""
        n_neurons = 8
        n_steps = 500
        i_values = [64, 96, 128, 160, 192, 224, 100, 140]
        currents = np.array(i_values, dtype=np.int16)

        sequential_spikes = []
        for i_t in i_values:
            s, _ = v3.batch_lif_run(n_steps, 20, 256, i_t)
            sequential_spikes.append(np.asarray(s))

        par_spikes, _ = v3.batch_lif_run_multi(n_neurons, n_steps, 20, 256, currents)
        par_arr = np.asarray(par_spikes)

        for ni in range(n_neurons):
            np.testing.assert_array_equal(
                par_arr[ni], sequential_spikes[ni], err_msg=f"Neuron {ni} mismatch"
            )

    def test_deterministic(self) -> None:
        """Same inputs -> same outputs."""
        currents = np.full(4, 128, dtype=np.int16)
        s1, v1 = v3.batch_lif_run_multi(4, 100, 20, 256, currents)
        s2, v2 = v3.batch_lif_run_multi(4, 100, 20, 256, currents)
        np.testing.assert_array_equal(np.asarray(s1), np.asarray(s2))
        np.testing.assert_array_equal(np.asarray(v1), np.asarray(v2))

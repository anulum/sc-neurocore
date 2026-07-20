# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — v3 LIF-kernel contracts

"""Contracts for sequential, batch and zero-allocation v3 LIF kernels."""

from __future__ import annotations

import numpy as np
import sc_neurocore_engine as v3


class TestBranchlessLIF:
    """Test that branchless LIF step produces identical results."""

    def test_100_steps_constant_input(self) -> None:
        """Standard equivalence: same as equivalence suite."""
        lif = v3.FixedPointLif()
        results = []
        for _ in range(100):
            s, v = lif.step(20, 256, 128, 0)
            results.append((s, v))
        assert len(results) == 100
        for s, v in results:
            assert s in (0, 1)
            assert isinstance(v, (int, np.integer))

    def test_batch_matches_step_by_step(self) -> None:
        """batch_lif_run must match step-by-step execution."""
        lif = v3.FixedPointLif()
        step_spikes, step_voltages = [], []
        for _ in range(1000):
            s, v = lif.step(20, 256, 128, 0)
            step_spikes.append(s)
            step_voltages.append(v)

        batch_spikes, batch_voltages = v3.batch_lif_run(1000, 20, 256, 128)
        np.testing.assert_array_equal(step_spikes, np.asarray(batch_spikes))
        np.testing.assert_array_equal(step_voltages, np.asarray(batch_voltages))

    def test_refractory_period(self) -> None:
        """Refractory behavior preserved under branchless mask."""
        spikes, _ = v3.batch_lif_run(200, 20, 256, 200, refractory_period=5)
        spikes_arr = np.asarray(spikes)
        spike_indices = np.where(spikes_arr == 1)[0]
        for idx in spike_indices:
            for ref_step in range(1, 6):
                if idx + ref_step < len(spikes_arr):
                    assert spikes_arr[idx + ref_step] == 0, (
                        f"Spike during refractory at step {idx + ref_step}"
                    )


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


class TestZeroAllocLIF:
    """Verify pre-allocated LIF batch outputs stay correct."""

    def test_batch_lif_unchanged(self) -> None:
        lif = v3.FixedPointLif()
        step_spikes, step_voltages = [], []
        for _ in range(1000):
            s, v = lif.step(20, 256, 128, 0)
            step_spikes.append(s)
            step_voltages.append(v)

        batch_spikes, batch_voltages = v3.batch_lif_run(1000, 20, 256, 128)
        np.testing.assert_array_equal(step_spikes, np.asarray(batch_spikes))
        np.testing.assert_array_equal(step_voltages, np.asarray(batch_voltages))

    def test_batch_lif_multi_unchanged(self) -> None:
        n_steps = 500
        currents = np.array([64, 96, 128, 160, 192, 224, 100, 140], dtype=np.int16)
        sequential_spikes = []
        for i_t in currents:
            spikes, _ = v3.batch_lif_run(n_steps, 20, 256, int(i_t))
            sequential_spikes.append(np.asarray(spikes))

        spikes_multi, _ = v3.batch_lif_run_multi(len(currents), n_steps, 20, 256, currents)
        spikes_multi = np.asarray(spikes_multi)
        for idx in range(len(currents)):
            np.testing.assert_array_equal(spikes_multi[idx], sequential_spikes[idx])

    def test_batch_lif_multi_shape(self) -> None:
        currents = np.full(10, 128, dtype=np.int16)
        spikes, voltages = v3.batch_lif_run_multi(10, 100, 20, 256, currents)
        spikes_arr = np.asarray(spikes)
        voltages_arr = np.asarray(voltages)
        assert spikes_arr.shape == (10, 100)
        assert voltages_arr.shape == (10, 100)
        assert spikes_arr.dtype == np.int32
        assert voltages_arr.dtype == np.int16

    def test_batch_lif_varying_unchanged(self) -> None:
        currents = np.array([120, 128, 136, 150, 160, 100, 80, 140], dtype=np.int16)
        noises = np.array([0, 1, -1, 2, -2, 0, 1, -1], dtype=np.int16)

        lif = v3.FixedPointLif()
        ref_spikes, ref_voltages = [], []
        for i_t, n_t in zip(currents, noises):
            s, v = lif.step(20, 256, int(i_t), int(n_t))
            ref_spikes.append(s)
            ref_voltages.append(v)

        spikes, voltages = v3.batch_lif_run_varying(
            leak_k=20,
            gain_k=256,
            currents=currents,
            noises=noises,
        )
        np.testing.assert_array_equal(np.asarray(spikes), np.array(ref_spikes, dtype=np.int32))
        np.testing.assert_array_equal(np.asarray(voltages), np.array(ref_voltages, dtype=np.int16))

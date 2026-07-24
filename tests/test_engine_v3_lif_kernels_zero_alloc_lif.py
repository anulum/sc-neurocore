# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestZeroAllocLIF from former test_engine_v3_lif_kernels.py

"""Focused suite: TestZeroAllocLIF from former test_engine_v3_lif_kernels.py."""

from __future__ import annotations

from tests.engine_v3_lif_kernels_support import *  # noqa: F403


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

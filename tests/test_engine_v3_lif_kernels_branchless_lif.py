# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestBranchlessLIF from former test_engine_v3_lif_kernels.py

"""Focused suite: TestBranchlessLIF from former test_engine_v3_lif_kernels.py."""

from __future__ import annotations

from tests.engine_v3_lif_kernels_support import *  # noqa: F403


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

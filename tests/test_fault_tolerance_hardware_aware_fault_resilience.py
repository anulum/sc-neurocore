# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestHardwareAwareFaultResilience from former test_fault_tolerance.py

"""Focused suite: TestHardwareAwareFaultResilience from former test_fault_tolerance.py."""

from __future__ import annotations

from tests.fault_tolerance_support import *  # noqa: F403

class TestHardwareAwareFaultResilience:
    """Test that hardware-aware training builds fault resilience."""

    def test_stuck_synapses_dont_degrade_output(self):
        """Network with stuck synapses should still produce valid output."""
        layer = HardwareAwareSCLayer(n_inputs=8, n_neurons=4, length=128, stuck_rate=0.2, seed=42)
        out = layer.forward([0.5] * 8)
        assert out.shape == (4,)
        assert np.all(np.isfinite(out))

    def test_increasing_stuck_rate_degrades_gracefully(self):
        """Higher stuck rates should still produce finite outputs."""
        results = {}
        for rate in [0.0, 0.1, 0.3, 0.5]:
            layer = HardwareAwareSCLayer(
                n_inputs=8, n_neurons=4, length=256, stuck_rate=rate, seed=42
            )
            out = layer.forward([0.5] * 8)
            results[rate] = np.mean(out)
        # All outputs should be finite and non-negative
        for rate, mean_out in results.items():
            assert np.isfinite(mean_out), f"rate={rate}: mean_out={mean_out}"
            assert mean_out >= 0.0, f"rate={rate}: mean_out={mean_out}"

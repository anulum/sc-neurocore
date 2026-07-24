# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestIntegration from former test_arcane_zenith.py

"""Focused suite: TestIntegration from former test_arcane_zenith.py."""

from __future__ import annotations

from tests.test_arcane_zenith.arcane_zenith_support import *  # noqa: F403


class TestIntegration:
    def test_long_run_keeps_all_meta_parameters_bounded(self):
        """Across 1000 steps of varied input, the four meta-parameters must
        stay strictly inside the biological ranges the module documents.
        Any escape indicates either a broken clamp in ``_map_to_range`` or
        an out-of-bounds weight leaking from the plasticity rule."""
        core = create_arcane_neuron_with_zenith_plasticity(backend="torch")
        rng = np.random.default_rng(seed=20260420)
        for _ in range(1000):
            current = float(rng.uniform(-5.0, 10.0))
            core.step(current)
            assert 1000.0 <= core.neuron.tau_deep <= 50000.0
            assert 0.01 <= core.neuron.surprise_baseline <= 0.5
            assert 0.0 <= core.neuron.delta_conf <= 1.0
            assert 0.001 <= core.neuron.lr_base <= 0.1

    def test_identity_drift_monotonic_non_decreasing(self):
        """``identity_drift`` accumulates |Δv_deep|, so it may only ever
        grow (or stay flat when v_deep is stationary)."""
        core = create_arcane_neuron_with_zenith_plasticity(backend="torch")
        drifts = [core.neuron.identity_drift]
        for _ in range(200):
            core.step(3.0)
            drifts.append(core.neuron.identity_drift)
        assert all(drifts[i + 1] >= drifts[i] for i in range(len(drifts) - 1))

# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestArcaneIdentityPersistence from former test_model_arcane_neuron.py

"""Focused suite: TestArcaneIdentityPersistence from former test_model_arcane_neuron.py."""

from __future__ import annotations

from tests.model_arcane_neuron_support import *  # noqa: F403

class TestArcaneIdentityPersistence:
    """CORE: v_deep is the identity — it PERSISTS through reset."""

    def test_deep_persists_through_reset(self):
        """reset() zeros v_fast and v_work but NOT v_deep."""
        n = ArcaneNeuron()
        for _ in range(10000):
            n.step(2.0)
        deep_before = n.v_deep
        assert deep_before > 0, "v_deep should accumulate during firing"
        n.reset()
        assert n.v_deep == deep_before, f"v_deep changed from {deep_before} to {n.v_deep} on reset"
        assert n.v_fast == 0.0 and n.v_work == 0.0

    def test_deep_accumulates_slowly(self):
        """v_deep changes on tau_deep=10000 timescale — ultra-slow."""
        n = ArcaneNeuron()
        for _ in range(100):
            n.step(2.0)
        d100 = n.v_deep
        for _ in range(10000):
            n.step(2.0)
        d10k = n.v_deep
        assert abs(d10k) > abs(d100), "v_deep should grow over long runs"
        assert abs(d10k) < 0.1, f"v_deep = {d10k} — should be small (tau=10k)"

    def test_deep_requires_novelty(self):
        """v_deep updates proportional to novelty: dv_deep ∝ v_work * novelty."""
        n = ArcaneNeuron()
        n.alpha_d = 0.0  # disable deep accumulation
        for _ in range(10000):
            n.step(2.0)
        assert abs(n.v_deep) < 1e-10, "alpha_d=0 should prevent deep update"

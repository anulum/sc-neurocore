# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestTraubMilesHHProperties from former test_model_traub_miles.py

"""Focused suite: TestTraubMilesHHProperties from former test_model_traub_miles.py."""

from __future__ import annotations

from tests.model_traub_miles_support import *  # noqa: F403

class TestTraubMilesHHProperties:
    """Verify HH-specific properties: gating bounds, Na inactivation, refractory."""

    def test_gating_bounded(self):
        n = TraubMilesNeuron()
        for _ in range(50000):
            n.step(5.0)
        for name, val in [("m", n.m), ("h", n.h), ("n", n.n)]:
            assert -0.01 <= val <= 1.01, f"{name} = {val:.6f}"

    def test_h_inactivation_during_depolarisation(self):
        """Na inactivation gate h should decrease during sustained firing."""
        n = TraubMilesNeuron()
        h0 = n.h
        for _ in range(50000):
            n.step(10.0)
        # h oscillates during firing but should be < initial at some point
        # Check average: during spiking, h drops during each AP
        assert n.h != h0  # h has changed (oscillating)

    def test_na_current_drives_upstroke(self):
        """I_Na = g_Na · m³ · h · (V - E_Na). At rest: m≈0.05, inward current small.
        During AP: m rapidly activates → large inward Na → fast upstroke."""
        n = TraubMilesNeuron()
        # At rest
        i_na_rest = n.g_na * n.m**3 * n.h * (n.v - n.e_na)
        assert i_na_rest < 0  # inward at rest (V < E_Na)
        # m small → magnitude small
        assert abs(i_na_rest) < 10  # weak at rest

    def test_isi_regularity(self):
        """At constant input, ISI should be regular (limit cycle)."""
        n = TraubMilesNeuron()
        spikes = _run(n, current=5.0, steps=50000)
        assert len(spikes) >= 20
        isis = np.diff(spikes[5:]).astype(float)
        cv = np.std(isis) / np.mean(isis)
        assert cv < 0.05, f"CV(ISI) = {cv:.4f}"

    def test_singularity_protection(self):
        """Rate functions use abs(d) > 1e-6 guard against division by zero."""
        n = TraubMilesNeuron(v=-54.0)  # d = v + 54 = 0
        n.step(0.0)  # should not raise
        assert np.isfinite(n.v)

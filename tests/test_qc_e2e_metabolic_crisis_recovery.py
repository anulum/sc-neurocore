# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestMetabolicCrisisRecovery from former test_qc_e2e.py

"""Focused suite: TestMetabolicCrisisRecovery from former test_qc_e2e.py."""

from __future__ import annotations

from tests.qc_e2e_support import *  # noqa: F403


class TestMetabolicCrisisRecovery:
    """Drain ATP, verify neurons recover and resume spiking."""

    def test_atp_recovery_after_depletion(self) -> None:
        pool = SpinPoolMPS(n_sites=8)
        neurons = [HybridFisherPosnerLIF(i, pool) for i in range(8)]

        # Deplete all ATP
        for n in neurons:
            n.atp_level = 0.0

        # Verify metabolic failure occurs
        failures_before = sum(n._metabolic_failures for n in neurons)
        for _ in range(10):
            for n in neurons:
                n.step(100.0)  # strong input, but no ATP
        failures_after = sum(n._metabolic_failures for n in neurons)
        assert failures_after > failures_before, "Should accumulate metabolic failures"

        # Let neurons recover (500 steps with no input)
        for _ in range(500):
            for n in neurons:
                n.step(0.0)

        # ATP should have partially recovered
        avg_atp = np.mean([n.atp_level for n in neurons])
        assert avg_atp > 0.1, f"ATP should recover: avg={avg_atp}"

        # Resume strong input — should eventually spike
        spikes = 0
        for _ in range(200):
            for n in neurons:
                _, spiked = n.step(100.0)
                if spiked:
                    spikes += 1
        assert spikes > 0, "Neurons should resume spiking after recovery"

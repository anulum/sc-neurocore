# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (surrogate_feed) from former test_observation_loader.py

from __future__ import annotations

from observation_loader_support import *  # noqa: F403

def test_loaded_observation_feeds_surrogate_optimizer() -> None:
    observations = observations_from_payload(
        {
            "observations": [
                {
                    **_design(),
                    "luts_used": 260,
                    "power_mw": 1.1,
                    "latency_cycles": 128,
                    "accuracy_score": 0.999,
                }
            ]
        }
    )
    target = TargetHardwareProfile(
        name="loader-integration",
        budget=HardwareBudget(max_luts=10_000, max_power_mw=100.0, max_latency_cycles=256),
    )
    optimiser = SurrogateSCOptimizer(
        target,
        bitstream_options=(64, 128),
        precision_options=(4, 8),
        observations=observations,
    )

    report = optimiser.optimise([LayerProfile("encoder", 256, is_critical_path=True)])

    assert report is not None
    assert report.feasible
    assert report.config["encoder"].bitstream_length == 128
    assert report.config["encoder"].precision_bits == 8

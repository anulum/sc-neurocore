# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestOptimizationResult from former test_optimizer_resource.py

"""Focused suite: TestOptimizationResult from former test_optimizer_resource.py."""

from __future__ import annotations

from tests.optimizer_resource_support import *  # noqa: F403

class TestOptimizationResult:
    def test_summary(self):
        r = OptimizationResult(
            fits=True,
            target="ice40",
            final_luts=1000,
            target_luts=5000,
            utilization_pct=20.0,
            final_bitstream_length=64,
            final_sparsity=0.3,
            steps=[
                OptimizationStep(
                    action="Reduce L to 128",
                    luts_before=6000,
                    luts_after=3000,
                    sparsity=0.0,
                    bitstream_length=128,
                ),
            ],
        )
        s = r.summary()
        assert "ice40" in s
        assert "YES" in s
        assert "1,000" in s

    def test_summary_does_not_fit(self):
        r = OptimizationResult(
            fits=False,
            target="ice40",
            final_luts=9000,
            target_luts=5000,
            utilization_pct=180.0,
            final_bitstream_length=256,
            final_sparsity=0.0,
        )
        s = r.summary()
        assert "NO" in s

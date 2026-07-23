# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestConfiguration from former test_uvm_gen.py

"""Focused suite: TestConfiguration from former test_uvm_gen.py."""

from __future__ import annotations

from uvm_gen_support import *  # noqa: F403

class TestConfiguration:
    def test_custom_stimulus(self):
        stim = StimulusConfig(num_transactions=500, bitstream_density_range=(0.2, 0.8))
        gen = UVMGenerator(stimulus=stim)
        bench = gen.generate(lif_module())
        assert "500" in bench.sequence_sv

    def test_custom_coverage(self):
        cov = CoverageSpec(bitstream_density_bins=20)
        gen = UVMGenerator(coverage=cov)
        bench = gen.generate(lif_module())
        assert "20" in bench.coverage_sv

    def test_scoreboard_config(self):
        sb = ScoreboardConfig(check_popcount=True, check_spike_timing=True)
        gen = UVMGenerator(scoreboard=sb)
        bench = gen.generate(lif_module())
        assert "popcount" in bench.scoreboard_sv or "spike" in bench.scoreboard_sv

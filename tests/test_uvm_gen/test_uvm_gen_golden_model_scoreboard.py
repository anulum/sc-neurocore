# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestGoldenModelScoreboard from former test_uvm_gen.py

"""Focused suite: TestGoldenModelScoreboard from former test_uvm_gen.py."""

from __future__ import annotations

from uvm_gen_support import *  # noqa: F403


class TestGoldenModelScoreboard:
    def test_golden_comparison_enabled_with_explicit_reference(self):
        sb = ScoreboardConfig(
            check_golden_comparison=True,
            golden_expressions={"v_out": "txn.I_t"},
        )
        gen = UVMGenerator(scoreboard=sb)
        bench = gen.generate(lif_module())
        assert "golden_compute" in bench.scoreboard_sv
        assert "return txn.I_t;" in bench.scoreboard_sv

    def test_golden_vars_present(self):
        sb = ScoreboardConfig(
            check_golden_comparison=True,
            golden_expressions={"v_out": "txn.I_t"},
        )
        gen = UVMGenerator(scoreboard=sb)
        bench = gen.generate(lif_module())
        assert "expected_v_out" in bench.scoreboard_sv

    def test_mismatch_detection(self):
        sb = ScoreboardConfig(
            check_golden_comparison=True,
            golden_expressions={"v_out": "txn.I_t"},
        )
        gen = UVMGenerator(scoreboard=sb)
        bench = gen.generate(lif_module())
        assert "MISMATCH" in bench.scoreboard_sv

    def test_golden_comparison_requires_explicit_reference(self):
        sb = ScoreboardConfig(check_golden_comparison=True)
        gen = UVMGenerator(scoreboard=sb)
        with pytest.raises(ValueError, match="Missing golden reference expression"):
            gen.generate(lif_module())

    def test_golden_disabled(self):
        sb = ScoreboardConfig(check_golden_comparison=False)
        gen = UVMGenerator(scoreboard=sb)
        bench = gen.generate(lif_module())
        assert "golden_compute" not in bench.scoreboard_sv

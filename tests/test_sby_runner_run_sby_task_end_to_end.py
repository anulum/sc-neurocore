# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestRunSbyTaskEndToEnd from former test_sby_runner.py

"""Focused suite: TestRunSbyTaskEndToEnd from former test_sby_runner.py."""

from __future__ import annotations

from tests.sby_runner_support import *  # noqa: F403

@_needs_formal
class TestRunSbyTaskEndToEnd:
    """A real ``sby`` task with the toolchain present."""

    def test_trivial_true_assertion_passes(self, tmp_path: Path) -> None:
        (tmp_path / "m.v").write_text(
            "module m(input wire clk);\n"
            "  reg [3:0] c = 0;\n"
            "  always @(posedge clk) begin c <= c + 1; assert (c <= 15); end\n"
            "endmodule\n",
            encoding="utf-8",
        )
        (tmp_path / "m.sby").write_text(
            "[tasks]\nbmc\n[options]\nbmc: mode bmc\nbmc: depth 6\n"
            "[engines]\nsmtbmc z3\n[script]\nread -formal m.v\nprep -top m\n[files]\nm.v\n",
            encoding="utf-8",
        )
        run = run_sby_task(tmp_path, "m.sby", timeout_s=60.0)
        assert run.verdict == "PASS"
        assert run.returncode == 0

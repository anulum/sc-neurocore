# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestCli from former test_gen_vmin_lif_lut.py

"""Focused suite: TestCli from former test_gen_vmin_lif_lut.py."""

from __future__ import annotations

from gen_vmin_lif_lut_support import *  # noqa: F403


class TestCli:
    def test_print_lut_outputs_table(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert main(["--print-lut"]) == 0

        output = capsys.readouterr().out
        assert "# Vmin_LIF softplus LUT" in output
        assert "q88=" in output

    def test_out_vh_writes_split_header(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        out_path = tmp_path / "vmin_lif_lut.vh"

        assert main(["--out-vh", str(out_path)]) == 0

        output = capsys.readouterr().out
        header = out_path.read_text(encoding="utf-8")
        assert "Written 64 LUT entries" in output
        assert "// SPDX-License-Identifier: AGPL-3.0-or-later\n" in header
        assert "// Commercial license available\n" in header

    def test_demo_outputs_trajectory(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert main(["--demo"]) == 0

        output = capsys.readouterr().out
        assert "Demo: 20 steps" in output
        assert "v_float" in output

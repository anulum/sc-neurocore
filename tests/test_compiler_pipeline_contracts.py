# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for compiler pipeline contracts

"""Contracts for compiler pipeline filesystem, tool, and path boundaries."""

from __future__ import annotations

import os
import tempfile

import pytest

from sc_neurocore.compiler.pipeline import CompilerPipeline
from sc_neurocore.exceptions import SCCompilerError


def test_compiler_pipeline_creates_nested_work_directory() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        work_dir = os.path.join(tmp, "sub", "dir")

        CompilerPipeline(work_dir=work_dir)

        assert os.path.isdir(work_dir)


def test_compiler_pipeline_fails_closed_when_firtool_does_not_lower() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        pipeline = CompilerPipeline(work_dir=tmp)

        with pytest.raises(SCCompilerError, match="firtool failed"):
            pipeline.compile_mlir_to_verilog("module test();", output_name="no_stub")

        assert not os.path.exists(os.path.join(tmp, "no_stub.v"))


def test_compiler_pipeline_uses_real_firtool_output(monkeypatch) -> None:
    def fake_firtool(cmd, check):
        assert check is True
        assert cmd[0] == "firtool"
        out_path = cmd[cmd.index("-o") + 1]
        with open(out_path, "w") as handle:
            handle.write("module real_lowered(); endmodule\n")

    with tempfile.TemporaryDirectory() as tmp:
        monkeypatch.setattr("subprocess.run", fake_firtool)
        pipeline = CompilerPipeline(work_dir=tmp)

        verilog_path = pipeline.compile_mlir_to_verilog("module test();", output_name="real")

        with open(verilog_path) as handle:
            lowered = handle.read()
        assert "module real_lowered" in lowered
        assert "Stub" not in lowered


def test_compiler_pipeline_rejects_empty_or_escaping_names() -> None:
    pipeline = CompilerPipeline(work_dir="/tmp/safe")

    with pytest.raises(ValueError, match="Invalid output name"):
        pipeline._sanitize_name("!!!")
    with pytest.raises(ValueError, match="Path escapes work_dir"):
        pipeline._validate_path("/etc/passwd")


def test_compiler_pipeline_rejects_unknown_synthesis_target() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        pipeline = CompilerPipeline(work_dir=tmp)
        verilog_path = os.path.join(tmp, "dummy.v")
        with open(verilog_path, "w") as handle:
            handle.write("module dummy(); endmodule")

        with pytest.raises(ValueError, match="Unknown target FPGA"):
            pipeline.run_synthesis(verilog_path, target_fpga="nope")

# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for compiler pipeline contracts

"""Contracts for compiler pipeline filesystem, tool, and path boundaries."""

from __future__ import annotations

import logging
import os
import subprocess
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


def test_compiler_pipeline_deletes_partial_verilog_on_firtool_failure(monkeypatch) -> None:
    """A partial Verilog file left behind by a failed firtool run is removed."""

    def fake_firtool(cmd, check):
        out_path = cmd[cmd.index("-o") + 1]
        with open(out_path, "w") as handle:
            handle.write("// partial broken output\n")
        raise subprocess.CalledProcessError(1, cmd)

    with tempfile.TemporaryDirectory() as tmp:
        monkeypatch.setattr("subprocess.run", fake_firtool)
        pipeline = CompilerPipeline(work_dir=tmp)
        partial = os.path.join(pipeline.work_dir, "broken.v")

        with pytest.raises(SCCompilerError, match="firtool failed"):
            pipeline.compile_mlir_to_verilog("module test();", output_name="broken")

        assert not os.path.exists(partial)


def test_compiler_pipeline_tolerates_missing_yosys(monkeypatch, caplog) -> None:
    """A missing or failing yosys is logged, not raised, and the json path is returned."""

    def fake_yosys(cmd, check):
        raise FileNotFoundError("yosys")

    with tempfile.TemporaryDirectory() as tmp:
        monkeypatch.setattr("subprocess.run", fake_yosys)
        pipeline = CompilerPipeline(work_dir=tmp)
        v_path = os.path.join(pipeline.work_dir, "design.v")

        with caplog.at_level(logging.WARNING):
            json_path = pipeline.run_synthesis(v_path, target_fpga="ice40")

        assert json_path == os.path.join(pipeline.work_dir, "design.json")
        assert any("yosys failed or not found" in record.message for record in caplog.records)


def test_compiler_pipeline_tolerates_missing_nextpnr(monkeypatch, caplog) -> None:
    """A missing or failing nextpnr is logged, not raised, and the asc path is returned."""

    def fake_nextpnr(cmd, check):
        raise subprocess.CalledProcessError(127, cmd)

    with tempfile.TemporaryDirectory() as tmp:
        monkeypatch.setattr("subprocess.run", fake_nextpnr)
        pipeline = CompilerPipeline(work_dir=tmp)
        json_path = os.path.join(pipeline.work_dir, "design.json")

        with caplog.at_level(logging.WARNING):
            asc_path = pipeline.run_pnr(json_path, target_device="up5k")

        assert asc_path == os.path.join(pipeline.work_dir, "design.asc")
        assert any("nextpnr failed or not found" in record.message for record in caplog.records)

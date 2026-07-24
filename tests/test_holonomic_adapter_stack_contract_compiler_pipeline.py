# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (compiler_pipeline) from former test_holonomic_adapter_stack_contract.py

from __future__ import annotations

from tests.holonomic_adapter_stack_contract_support import *  # noqa: F403

def test_compiler_pipeline_invokes_real_lowering(monkeypatch):
    def fake_tool(cmd, check, capture_output=False, text=False):
        assert check is True
        if cmd[0] == "circt-opt":
            # compile_mlir_to_verilog runs circt-opt --export-verilog and captures
            # the exported Verilog from stdout (the -o sink is discarded).
            assert "--export-verilog" in cmd
            return subprocess.CompletedProcess(
                cmd, 0, stdout="module test(); endmodule\n", stderr=""
            )
        if cmd[0] == "yosys":
            assert "-s" in cmd
        elif cmd[0] == "nextpnr-ice40":
            assert "--json" in cmd
            assert "--asc" in cmd
        else:
            raise AssertionError(f"unexpected tool command: {cmd}")
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr("subprocess.run", fake_tool)
    # The hardened pipeline resolves each tool via ``shutil.which`` before invoking
    # it; on a runner without the EDA toolchain (e.g. CI, which ships no firtool)
    # resolution raises before the mocked ``subprocess.run`` is reached. Stub
    # resolution to the bare tool name so the command-construction contract below
    # is still exercised and ``fake_tool`` matches on ``cmd[0]``.
    monkeypatch.setattr(CompilerPipeline, "_resolve_tool", staticmethod(lambda name: name))
    pipeline = CompilerPipeline(work_dir=".tmp/test_compiler")
    mlir = "hw.module @test() { hw.output }"
    v_path = pipeline.compile_mlir_to_verilog(mlir, "test")
    assert v_path.endswith(".v")
    json_path = pipeline.run_synthesis(v_path)
    assert json_path.endswith(".json")
    asc_path = pipeline.run_pnr(json_path)
    assert asc_path.endswith(".asc")



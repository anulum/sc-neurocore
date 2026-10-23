# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — CIRCT numerical gate refusal regressions

"""Refuse vacuous corpora, incomplete inputs and corrupted/missing observation traces."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pytest

from tools.check_ir_circt import validate_observations


def test_cli_refuses_empty_native_corpus(tmp_path: Path) -> None:
    """An empty native run cannot produce a green CIRCT validation result."""
    tool = Path(__file__).resolve().parents[2] / "tools/check_ir_circt.py"
    result = subprocess.run(
        [sys.executable, str(tool), str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert result.returncode == 1
    assert "no CIRCT cases" in result.stderr


def test_cli_refuses_incomplete_circuit_before_tool_execution(tmp_path: Path) -> None:
    """Missing testbench/reference inputs fail even if a compiler is unavailable."""
    case = tmp_path / "incomplete"
    case.mkdir()
    (case / "graph.mlir").write_text("", encoding="utf-8")
    tool = Path(__file__).resolve().parents[2] / "tools/check_ir_circt.py"
    result = subprocess.run(
        [sys.executable, str(tool), str(tmp_path), "--circt-opt", "absent-circt-tool"],
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert result.returncode == 1
    assert "tb.sv" in result.stderr


@pytest.mark.parametrize(
    "trace", ["", "OBS 1\n", "OBS 2\nOBS 1\n", "OBS 1\nOBS x\n", "OBS 1\nOBS 2\nOBS 2\n"]
)
def test_inexact_or_incomplete_observations_are_errors(trace: str) -> None:
    """Preserve sample count, order, unknown values and repeated completion samples."""
    with pytest.raises(ValueError, match="reference mismatch"):
        validate_observations(["OBS 1", "OBS 2"], trace)


def test_simulator_finish_metadata_does_not_change_exact_observations() -> None:
    """Tool metadata is distinct from the circuit's complete declared observations."""
    assert validate_observations(["OBS 1", "OBS 2"], "OBS 1\nOBS 2\ntb.sv: $finish\n") == 2


@pytest.mark.parametrize(
    ("metadata", "diagnostic"),
    [
        ("[]", "requires exactly"),
        ('{"schema_version":"other","kind":"overflow"}', "unsupported runtime failure"),
        (
            '{"schema_version":"sc-ir-runtime-failure.v1","kind":"unknown"}',
            "unknown runtime failure",
        ),
        ('{"schema_version":"sc-ir-runtime-failure.v1","kind":false}', "unknown runtime failure"),
        (
            '{"schema_version":"sc-ir-runtime-failure.v1","kind":"overflow","extra":1}',
            "requires exactly",
        ),
        (
            '{"schema_version":"sc-ir-runtime-failure.v1","kind":"domain","kind":"overflow"}',
            "duplicate runtime failure",
        ),
    ],
)
def test_cli_refuses_ambiguous_runtime_contract_before_compilation(
    metadata: str, diagnostic: str, tmp_path: Path
) -> None:
    """Reject malformed failure contracts around a real native-generated checked circuit."""
    from sc_neurocore_engine.ir import ScGraphBuilder

    b = ScGraphBuilder("failure_contract")
    valid = b.input("valid", "bool")
    b.runtime_check(valid)
    b.output("result", valid)
    graph = b.build()
    case = tmp_path / "checked"
    case.mkdir()
    (case / "graph.mlir").write_text(graph.emit_mlir(), encoding="utf-8")
    (case / "tb.sv").write_text(
        "module tb; wire result; failure_contract dut(.clk(1'b0),.rst_n(1'b0),"
        '.valid(1\'b1),.result(result)); initial begin #1; $display("OBS %h",result); '
        "$finish; end endmodule",
        encoding="utf-8",
    )
    (case / "expected.txt").write_text("OBS 1\n", encoding="utf-8")
    (case / "runtime_failure.json").write_text(metadata, encoding="utf-8")
    tool = Path(__file__).resolve().parents[2] / "tools/check_ir_circt.py"
    report = tmp_path / "report.json"
    actual = subprocess.run(
        [
            sys.executable,
            str(tool),
            str(tmp_path),
            "--circt-opt",
            "absent-circt-tool",
            "--report",
            str(report),
        ],
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert actual.returncode == 1
    assert diagnostic in actual.stderr
    assert "No such file or directory" not in actual.stderr
    assert not report.exists()

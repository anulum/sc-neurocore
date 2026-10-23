# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Runtime check native API and real hardware simulation

"""Exercise range failures, controls, persistence and synthesis through actual native graphs."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess

import pytest

from tests.engine_requirement import require_engine

require_engine()

from sc_neurocore_engine.ir import ScGraphBuilder, parse_ir


@pytest.mark.parametrize(
    "case",
    [
        "positive-range",
        "disabled-fault",
        "overflow-fault",
        "domain-fault",
        "constant-pass",
        "constant-fault",
        "effect-only-fault",
    ],
)
def test_runtime_check_actual_simulation_and_synthesis(case: str, tmp_path: Path) -> None:
    """Require real fatals only on invalid enabled values and preserve all synthesis data."""
    constant = case.startswith("constant-") or case == "effect-only-fault"
    kind = "domain" if case == "domain-fault" else "overflow"
    b = ScGraphBuilder("SYNTHESIS_MODEL")
    if constant:
        predicate = b.constant_i64(int(case == "constant-pass"), "bool")
        b.runtime_check(predicate, kind)
        if case == "effect-only-fault":
            bench = (
                "module tb; SYNTHESIS_MODEL dut(.clk(1'b0),.rst_n(1'b0)); "
                "initial begin #1; $finish; end endmodule"
            )
            full = []
        else:
            b.output("result", b.constant_i64(1, "bool"))
            bench = (
                "module tb; wire result; SYNTHESIS_MODEL dut(.clk(1'b0),.rst_n(1'b0),.result(result)); "
                'initial begin #1; $display("OBS %h",result); $finish; end endmodule'
            )
            full = ["OBS 1"]
    else:
        sample = b.input("sample", "i16")
        enabled = b.input("enabled", "bool")
        lower = b.word_compare("ge", sample, b.constant_i64(-128, "i16"))
        upper = b.word_compare("le", sample, b.constant_i64(127, "i16"))
        valid = b.word_select(lower, upper, b.constant_i64(0, "bool"), "bool")
        b.runtime_check(valid, kind, enabled)
        b.output("result", sample)
        values = {
            "positive-range": [(-128, 1), (127, 1), (0, 1)],
            "disabled-fault": [(128, 0), (-129, 0)],
            "overflow-fault": [(128, 1)],
            "domain-fault": [(-129, 1)],
        }[case]
        steps = []
        for value, enable in values:
            steps.append(
                f"sample=16'h{value & 65535:04x}; enabled=1'b{enable}; "
                '#1; $display("OBS %h",result); #1;'
            )
        bench = (
            "module tb; reg signed [15:0] sample=0; reg enabled=0; wire [15:0] result; "
            "SYNTHESIS_MODEL dut(.clk(1'b0),.rst_n(1'b0),.sample(sample),.enabled(enabled),.result(result)); "
            'initial begin #1; $display("OBS %h",result); #1; '
            + " ".join(steps)
            + " $finish; end endmodule"
        )
        full = ["OBS 0000", *(f"OBS {value & 65535:04x}" for value, _ in values)]
    failure = case.endswith("fault") and case != "disabled-fault"
    expected = ([] if constant else full[:1]) if failure else full
    graph = b.build()
    assert graph.verify() is None
    text = graph.to_text()
    restored = parse_ir(text)
    assert restored.to_text() == text
    source = tmp_path / "top.sv"
    source.write_text(restored.emit_sv(), encoding="utf-8")
    tb = tmp_path / "tb.sv"
    tb.write_text(bench, encoding="utf-8")
    for synthesis in [False, True]:
        binary = tmp_path / ("synthesis.vvp" if synthesis else "normal.vvp")
        command = ["iverilog", "-g2012", "-s", "tb"]
        if synthesis:
            command.append("-DSYNTHESIS")
        compiled = subprocess.run(
            [*command, "-o", str(binary), str(source), str(tb)],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        assert compiled.returncode == 0, compiled.stderr
        actual = subprocess.run(
            ["vvp", str(binary)], capture_output=True, text=True, timeout=30, check=False
        )
        observations = [line for line in actual.stdout.splitlines() if line.startswith("OBS ")]
        if failure and not synthesis:
            message = "SCIR arithmetic domain" if kind == "domain" else "SCIR overflow"
            assert actual.returncode == 1
            assert any(
                line.startswith("FATAL:") and message in line for line in actual.stdout.splitlines()
            )
            assert observations == expected
        else:
            assert actual.returncode == 0, actual.stdout + actual.stderr
            assert observations == full
    root = os.environ.get("SCN_IR_CIRCT_RECEIPT_DIR")
    if root:
        directory = Path(root) / case
        directory.mkdir(parents=True, exist_ok=False)
        (directory / "graph.scir").write_text(text, encoding="utf-8")
        (directory / "graph.mlir").write_text(restored.emit_mlir(), encoding="utf-8")
        (directory / "tb.sv").write_text(bench, encoding="utf-8")
        (directory / "expected.txt").write_text("\n".join(expected) + "\n", encoding="utf-8")
        (directory / "expected_synthesis.txt").write_text("\n".join(full) + "\n", encoding="utf-8")
        if failure:
            (directory / "runtime_failure.json").write_text(
                json.dumps({"schema_version": "sc-ir-runtime-failure.v1", "kind": kind}) + "\n",
                encoding="utf-8",
            )


@pytest.mark.parametrize("consumer", ["output", "condition", "enable"])
def test_runtime_check_effects_cannot_be_consumed_as_data(consumer: str) -> None:
    """Reject the real effect identifier before either backend emits an invalid circuit."""
    b = ScGraphBuilder("check_consumer")
    valid = b.input("valid", "bool")
    check = b.runtime_check(valid)
    if consumer == "output":
        b.output("result", check)
    else:
        b.runtime_check(
            check if consumer == "condition" else valid,
            enable=check if consumer == "enable" else None,
        )
        b.output("result", valid)
    graph = b.build()
    with pytest.raises(ValueError, match="not defined before use"):
        graph.emit_sv()
    with pytest.raises(ValueError, match="not defined before use"):
        graph.emit_mlir()


def test_runtime_check_refusals_preserve_builder_and_consumption_contract() -> None:
    """Reject unknown reasons and out-of-range IDs without allocating an effect."""
    b = ScGraphBuilder("check_recovery")
    valid = b.input("valid", "bool")
    with pytest.raises(ValueError, match="unknown runtime check kind"):
        b.runtime_check(valid, "unexpected")
    for value in [-1, 1 << 32]:
        with pytest.raises(OverflowError):
            b.runtime_check(value)
    assert b.runtime_check(valid) == valid + 1
    b.output("result", valid)
    assert b.build().verify() is None
    with pytest.raises(ValueError, match="consumed"):
        b.runtime_check(valid)

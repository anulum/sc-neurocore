# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Public native pipeline word register tests

"""Observe the public Python/native word-delay boundary through real RTL and reset."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from tests.engine_requirement import require_engine

require_engine()

from sc_neurocore_engine.ir import ScGraphBuilder, parse_ir


@pytest.mark.parametrize("width", [1, 16, 64, 65, 128, 131])
@pytest.mark.parametrize("signed", [False, True])
def test_public_word_delay_preserves_wide_bits_and_two_clock_history(
    width: int, signed: bool, tmp_path: Path
) -> None:
    """Zero reset wins between edges and while high; input changes never bypass either stage."""
    ty = f"{'i' if signed else 'u'}{width}"
    b = ScGraphBuilder("python_word_delay")
    source = b.input("sample", ty)
    first = b.word_delay(source, ty)
    second = b.word_delay(first, ty)
    b.output("first", first)
    b.output("second", second)
    graph = b.build()
    assert graph.verify() is None
    restored = parse_ir(graph.to_text())
    assert restored.to_text() == graph.to_text()
    assert restored.emit_sv() == graph.emit_sv()
    assert restored.emit_mlir() == graph.emit_mlir()
    digits, mask = (width + 3) // 4, (1 << width) - 1
    older = previous = 0
    expected = [f"OBS {previous:0{digits}x} {older:0{digits}x}"]
    stimuli = []
    values = [0, -1, 1 << (width - 1), 1, (1 << 63) | (1 << 64), 0, -17, 31]
    for index, value in enumerate(values):
        value &= mask
        stimuli.append(
            f'sample={width}\'h{value:x}; #1; $display("OBS %h %h",first,second); clk=1; #1; $display("OBS %h %h",first,second); clk=0; #1;'
        )
        expected.append(f"OBS {previous:0{digits}x} {older:0{digits}x}")
        older, previous = previous, value
        expected.append(f"OBS {previous:0{digits}x} {older:0{digits}x}")
        if index == 4:
            stimuli.append('#1; rst_n=0; #1; $display("OBS %h %h",first,second); rst_n=1; #1;')
            older = previous = 0
            expected.append(f"OBS {previous:0{digits}x} {older:0{digits}x}")
    stimuli.append(
        f'sample={width}\'h{mask:x}; rst_n=0; #1; clk=1; #1; $display("OBS %h %h",first,second); rst_n=1; #1; $display("OBS %h %h",first,second); clk=0; #1; clk=1; #1; $display("OBS %h %h",first,second);'
    )
    expected.extend([f"OBS {0:0{digits}x} {0:0{digits}x}"] * 2)
    expected.append(f"OBS {mask:0{digits}x} {0:0{digits}x}")
    bench = (
        f'module tb; reg clk=0,rst_n=1; reg [{width - 1}:0] sample=0; wire [{width - 1}:0] first,second; python_word_delay dut(.*); initial begin #1; rst_n=0; #1; $display("OBS %h %h",first,second); rst_n=1; #1;\n'
        + "\n".join(stimuli)
        + "\n$finish; end endmodule\n"
    )
    top, tb, binary = tmp_path / "top.sv", tmp_path / "tb.sv", tmp_path / "run.vvp"
    top.write_text(restored.emit_sv(), encoding="utf-8")
    tb.write_text(bench, encoding="utf-8")
    subprocess.run(
        ["iverilog", "-g2012", "-s", "tb", "-o", str(binary), str(top), str(tb)],
        check=True,
        capture_output=True,
        text=True,
        timeout=90,
    )
    observed = subprocess.run(
        ["vvp", str(binary)], check=True, capture_output=True, text=True, timeout=90
    ).stdout.splitlines()
    assert [line for line in observed if line.startswith("OBS ")] == expected
    if root := os.environ.get("SCN_IR_CIRCT_RECEIPT_DIR"):
        case = Path(root) / f"python-word-delay-{ty}"
        case.mkdir(parents=True, exist_ok=True)
        (case / "graph.scir").write_text(restored.to_text(), encoding="utf-8")
        (case / "graph.mlir").write_text(restored.emit_mlir(), encoding="utf-8")
        (case / "tb.sv").write_text(bench, encoding="utf-8")
        (case / "expected.txt").write_text("\n".join(expected) + "\n", encoding="utf-8")


@pytest.mark.parametrize(
    ("source_type", "result_type"),
    [
        ("i128", "u128"),
        ("u131", "u128"),
        ("bool", "bool"),
        ("rate", "rate"),
        ("fixed<16,8>", "fixed<16,8>"),
    ],
)
def test_public_word_delay_refuses_implicit_word_conversion(
    source_type: str, result_type: str
) -> None:
    """Native verification and both real emitters refuse mismatched or non-integer storage."""
    b = ScGraphBuilder("invalid_python_delay")
    x = b.input("sample", source_type)
    b.output("result", b.word_delay(x, result_type))
    graph = b.build()
    assert graph.verify()
    with pytest.raises(ValueError):
        graph.emit_sv()
    with pytest.raises(ValueError):
        graph.emit_mlir()

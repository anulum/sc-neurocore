# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Installed Python stream decoding through real RTL

"""Exercise the public Python graph builder, native bridge and emitted stream circuits."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from tests.engine_requirement import require_engine

require_engine()

from sc_neurocore_engine.ir import ScGraphBuilder, parse_ir


@pytest.mark.parametrize("window", [1, 3, 17, 64])
@pytest.mark.parametrize("lanes", [1, 3])
def test_public_decoder_matches_complete_rtl_windows(
    tmp_path: Path, window: int, lanes: int
) -> None:
    """Compare held count and Q8.8 rate after every rising clock edge."""
    b = ScGraphBuilder("python_stream")
    ty = f"bitstream<{window}>"
    if lanes > 1:
        ty = f"vec<{ty},{lanes}>"
    bits = b.input("bits", ty)
    b.output("count", b.popcount(bits))
    b.output("rate", b.decode(bits))
    graph = b.build()
    assert graph.verify() is None
    restored = parse_ir(graph.to_text())
    assert restored.to_text() == graph.to_text()
    assert restored.emit_sv() == graph.emit_sv()
    assert restored.emit_mlir() == graph.emit_mlir()
    top = tmp_path / "top.sv"
    top.write_text(restored.emit_sv(), encoding="utf-8")
    lines = [
        f"module tb; reg clk=0,rst_n=1; reg [{lanes - 1}:0] bits=0;",
        "wire [63:0] count; wire [15:0] rate; python_stream dut(.*);",
        "initial begin #1; rst_n=0; #1; rst_n=1;",
    ]
    expected: list[str] = []
    phase = partial = held = 0
    for i in range(window * 3 + 5):
        if i == window + 1:
            lines.append('#1; rst_n=0; #1; $display("OBS %0d %0d",count,rate); rst_n=1;')
            expected.append("OBS 0 0")
            phase = partial = held = 0
        word = (1 << lanes) - 1 if i < window * 2 else i % (1 << lanes)
        partial += word.bit_count()
        phase += 1
        if phase == window:
            held, phase, partial = partial, 0, 0
        expected.append(f"OBS {held} {held * 256 // (window * lanes)}")
        lines.append(
            f'bits={lanes}\'d{word}; #2; clk=1; #1; $display("OBS %0d %0d",count,rate); #2; clk=0;'
        )
    lines.append("$finish; end endmodule")
    tb = tmp_path / "tb.sv"
    tb.write_text("\n".join(lines), encoding="utf-8")
    binary = tmp_path / "stream.vvp"
    subprocess.run(
        ["iverilog", "-g2012", "-s", "tb", "-o", str(binary), str(top), str(tb)],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    simulation = subprocess.run(
        ["vvp", str(binary)], capture_output=True, text=True, check=True, timeout=30
    )
    observations = [line for line in simulation.stdout.splitlines() if line.startswith("OBS ")]
    assert observations == expected


@pytest.mark.parametrize("ty", ["bitstream<0>", "u0", "vec<bool,0>", "rate"])
def test_public_decoder_rejects_invalid_observation_shapes(ty: str) -> None:
    """Refuse invalid geometry through public verification and both emitters."""
    b = ScGraphBuilder("invalid_observations")
    bits = b.input("bits", ty)
    b.output("rate", b.decode(bits))
    graph = b.build()
    assert graph.verify()
    with pytest.raises(ValueError):
        graph.emit_sv()
    with pytest.raises(ValueError):
        graph.emit_mlir()


@pytest.mark.parametrize("width", [1, 64, 65, 129])
def test_public_integer_division_preserves_the_complete_dividend(
    tmp_path: Path, width: int
) -> None:
    """Divide the full input word before truncating the quotient to its u64 result."""
    builder = ScGraphBuilder("python_division")
    dividend = builder.input("dividend", f"u{width}")
    builder.output("quotient", builder.div_const(dividend, 3))
    graph = parse_ir(builder.build().to_text())
    assert graph.verify() is None
    source = tmp_path / "division.sv"
    source.write_text(graph.emit_sv(), encoding="utf-8")
    words = [0, (1 << width) - 1, 1 << (width - 1)]
    drive = "\n".join(
        f'dividend={width}\'h{word:x}; #1; $display("OBS %0d",quotient);' for word in words
    )
    bench = tmp_path / "tb.sv"
    bench.write_text(
        f"module tb; reg [{width - 1}:0] dividend=0; wire [63:0] quotient; "
        "python_division dut(.clk(1'b0),.rst_n(1'b1),.dividend(dividend),.quotient(quotient)); "
        f"initial begin {drive} $finish; end endmodule\n",
        encoding="utf-8",
    )
    binary = tmp_path / "division.vvp"
    subprocess.run(
        ["iverilog", "-g2012", "-s", "tb", "-o", str(binary), str(source), str(bench)],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    simulation = subprocess.run(
        ["vvp", str(binary)], capture_output=True, text=True, check=True, timeout=30
    )
    observations = [line for line in simulation.stdout.splitlines() if line.startswith("OBS ")]
    assert observations == [f"OBS {(word // 3) & ((1 << 64) - 1)}" for word in words]


@pytest.mark.parametrize("ty", ["u0", "u4294967295", "bitstream<3>", "rate"])
def test_public_integer_division_refuses_invalid_or_noninteger_words(ty: str) -> None:
    """Reject invalid geometry before either emitter constructs a division circuit."""
    builder = ScGraphBuilder("invalid_division")
    dividend = builder.input("dividend", ty)
    builder.output("quotient", builder.div_const(dividend, 3))
    graph = builder.build()
    assert graph.verify()
    with pytest.raises(ValueError):
        graph.emit_sv()
    with pytest.raises(ValueError):
        graph.emit_mlir()


def test_public_compile_demo_uses_explicit_decoder(tmp_path: Path) -> None:
    """Run the documented executable and compile both generated public examples."""
    root = Path(__file__).resolve().parents[1]
    execution = subprocess.run(
        [
            sys.executable,
            str(root / "examples/03_ir_compile_demo.py"),
            "--output-dir",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    assert "sc.decode" in execution.stdout
    assert "Verification FAILED" not in execution.stdout
    primitives = [
        root / "hdl" / name
        for name in (
            "sc_bitstream_encoder.v",
            "sc_bitstream_synapse.v",
            "sc_dotproduct_to_current.v",
            "sc_lif_neuron.v",
            "sc_dense_matrix_layer.v",
        )
    ]
    for top, source in (
        ("sc_synapse", "generated_synapse.sv"),
        ("sc_dense_layer", "generated_dense.sv"),
    ):
        subprocess.run(
            [
                "iverilog",
                "-g2012",
                "-s",
                top,
                "-o",
                str(tmp_path / f"{top}.vvp"),
                str(tmp_path / source),
                *map(str, primitives),
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )

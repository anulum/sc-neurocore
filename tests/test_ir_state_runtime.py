# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Public Python register transactions

"""Exercise register construction across the Python facade, native graph and real RTL."""

from __future__ import annotations

from pathlib import Path
import subprocess

import pytest

from tests.engine_requirement import require_engine

require_engine()

from sc_neurocore_engine.ir import ScGraphBuilder, parse_ir


def test_python_register_feedback_matches_independent_words(tmp_path: Path) -> None:
    """Swap signed Q-format states atomically, hold disabled cycles and reset between edges."""
    b = ScGraphBuilder("python_state")
    enabled = b.input("enabled", "bool")
    ai = b.constant_f64(-1.25, "fixed<16,8>")
    bi = b.constant_f64(0.375, "fixed<16,8>")
    a = b.state(ai, "fixed<16,8>")
    other = b.state(bi, "fixed<16,8>")
    b.store(a, other, enabled)
    b.store(other, a, enabled)
    b.output("a", a)
    b.output("b", other)
    graph = b.build()
    assert graph.verify() is None
    restored = parse_ir(graph.to_text())
    assert restored.to_text() == graph.to_text()
    assert restored.emit_sv() == graph.emit_sv()
    assert restored.emit_mlir() == graph.emit_mlir()
    assert "seq.firreg" in restored.emit_mlir()
    x, y = -320, 96
    expected = [f"OBS {x & 65535:04x} {y & 65535:04x}"]
    drives = []
    for advance in [True, True, False, False, True, False, True]:
        if advance:
            x, y = y, x
        expected.append(f"OBS {x & 65535:04x} {y & 65535:04x}")
        drives.append(
            f'enabled={int(advance)}; #2; clk=1; #1; $display("OBS %h %h",a,b); #2; clk=0;'
        )
    drives.append('enabled=0; #1; rst_n=0; #1; $display("OBS %h %h",a,b);')
    expected.append("OBS fec0 0060")
    source = tmp_path / "top.sv"
    source.write_text(restored.emit_sv(), encoding="utf-8")
    bench = tmp_path / "tb.sv"
    bench.write_text(
        "module tb; reg clk=0,rst_n=1,enabled=0; wire [15:0] a,b; python_state dut(.*); "
        'initial begin #1; rst_n=0; #1; $display("OBS %h %h",a,b); #1; rst_n=1;'
        + "\n".join(drives)
        + "$finish; end endmodule",
        encoding="utf-8",
    )
    binary = tmp_path / "state.vvp"
    subprocess.run(
        ["iverilog", "-g2012", "-s", "tb", "-o", str(binary), str(source), str(bench)],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    trace = subprocess.run(
        ["vvp", str(binary)],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    ).stdout
    assert [line for line in trace.splitlines() if line.startswith("OBS ")] == expected


def test_python_store_default_enable_and_consumed_builder() -> None:
    """Optional enable updates on every edge and normal consumed-builder errors survive."""
    b = ScGraphBuilder("unconditional")
    initial = b.constant_i64(1, "u8")
    with pytest.raises(ValueError, match="Unknown IR type"):
        b.state(initial, "unknown")
    state = b.state(initial, "u8")
    with pytest.raises(OverflowError):
        b.store(-1, state)
    with pytest.raises(OverflowError):
        b.store(state, state, -1)
    b.store(state, state)
    b.output("current", state)
    graph = b.build()
    assert graph.verify() is None
    assert "enable=none" in graph.to_text()
    with pytest.raises(ValueError, match="already consumed"):
        b.state(initial, "u8")
    with pytest.raises(ValueError, match="already consumed"):
        b.store(state, state)


@pytest.mark.parametrize(
    ("tail", "message"),
    [
        ("", "exactly one Store"),
        ("%2 = sc.store %1, %0, enable=none\n%3 = sc.store %1, %0, enable=none", "multiple Store"),
        ("%2 = sc.store %0, %0, enable=none", "must be a State"),
        ("%2 = sc.store %1, %0, enable=%0", "Bool type"),
        ("%2 = sc.constant 0 : i8\n%3 = sc.store %1, %2, enable=none", "exact state type"),
        ('%2 = sc.store %1, %0, enable=none\nsc.output "invalid" %2', "not defined before use"),
    ],
)
def test_python_state_refuses_invalid_transaction(tail: str, message: str) -> None:
    """Native verification and both emitters refuse malformed persisted transactions."""
    graph = parse_ir(
        "sc.graph @invalid {\n%0 = sc.constant 1 : u8\n%1 = sc.state %0 : u8\n" + tail + "\n}"
    )
    errors = graph.verify()
    assert errors is not None
    assert any(message in error for error in errors)
    for emit in [graph.emit_sv, graph.emit_mlir]:
        with pytest.raises(ValueError, match=message):
            emit()


def test_python_raw_rate_reset_survives_graph_persistence() -> None:
    """Persist an encoded Q8.8 reset word without interpreting it as real-valued 1.0."""
    b = ScGraphBuilder("python_raw_rate")
    initial = b.constant_i64(1, "rate")
    state = b.state(initial, "rate")
    b.store(state, state)
    b.output("current", state)
    graph = b.build()
    assert graph.verify() is None
    assert "sc.constant i64(1) : rate" in graph.to_text()
    restored = parse_ir(graph.to_text())
    assert restored.to_text() == graph.to_text()
    assert restored.emit_sv() == graph.emit_sv()
    assert restored.emit_mlir() == graph.emit_mlir()
    assert "16'sd1;" in restored.emit_sv()

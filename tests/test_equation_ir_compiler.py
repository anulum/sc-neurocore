# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Public equation compiler source and domain contracts

"""Exercise native public equation compilation, source custody and domain refusals."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from sc_neurocore.compiler import compile_to_ir
from sc_neurocore.neurons.equation_builder import EquationNeuron
from tests.engine_requirement import require_engine

require_engine()

from sc_neurocore_engine.ir import parse_ir


def test_public_equation_reports_native_scalar_port_width_limit() -> None:
    """Wider internal words do not imply unsupported wider native input/state ports."""
    with pytest.raises(NotImplementedError, match="ports require data_width <= 64"):
        compile_to_ir(EquationNeuron({"v": "I"}, dt=1.0), data_width=72, fraction=0)


@pytest.mark.parametrize(
    ("signed", "overflow", "error"),
    [
        (False, "wrap", NotImplementedError),
        (True, "unknown", ValueError),
    ],
)
def test_public_equation_rejects_unsupported_state_commit_modes(
    signed: bool, overflow: str, error: type[Exception]
) -> None:
    """Invalid requested policies fail through the actual model compiler before emission."""
    with pytest.raises(error):
        compile_to_ir(EquationNeuron({"v": "I"}, dt=0.5), signed=signed, overflow=overflow)


def test_public_equation_rejects_reserved_native_module_identifier() -> None:
    """Native graph diagnostics propagate to the authored model's public request."""
    with pytest.raises(ValueError, match="invalid equation IR"):
        compile_to_ir(EquationNeuron({"v": "I"}, dt=0.5), module_name="SYNTHESIS")


def test_native_equation_source_uses_initial_state_and_preserves_live_state() -> None:
    """Model identity and graph stay constant after live state and RNG counters advance."""
    neuron = EquationNeuron(
        {"v": "rand() + gain * I", "w": "v - w"},
        state={"v": 0.25, "w": -0.5},
        parameters={"gain": 0.5},
        threshold="v > 1",
        reset={"w": "v", "v": "0"},
        dt=0.5,
        rng_seed=(1 << 63) + 7,
        random_stream=(1 << 31) + 9,
    )
    first = compile_to_ir(neuron)
    neuron.step(0.75)
    live = dict(neuron.state)
    second = compile_to_ir(neuron)
    assert neuron.state == live
    assert first.graph.to_text() == second.graph.to_text()
    payload = first.graph.source_payload
    assert payload is not None
    source = json.loads(payload)
    assert source["schema"] == "sc-equation-source.v1"
    assert source["model"]["initial_state"] == [["v", 0.25], ["w", -0.5]]
    assert source["model"]["reset"] == [["w", "v"], ["v", "0"]]
    assert source["model"]["rng_seed"] == (1 << 63) + 7
    assert source["model"]["random_stream"] == (1 << 31) + 9
    assert source["numeric"]["function_lowering"] == "lut-v1"
    assert first.graph.source_sha256 == hashlib.sha256(payload.encode()).hexdigest()
    restored = parse_ir(first.graph.to_text())
    assert restored.source_payload == payload
    assert restored.emit_sv() == first.graph.emit_sv()
    assert restored.emit_mlir() == first.graph.emit_mlir()


@pytest.mark.parametrize(
    "expression",
    [
        "I / v",
        "I / tau",
        "I / v if v >= 0 else 0",
        "I / v if v < 1 else 0",
        "I / v if v <= 0 or I / v > 0 else 0",
        "I / v if v > I else 0",
        "I / v if v > -I else 0",
        "I / v if v > 1j else 0",
    ],
)
def test_native_equation_refuses_unproven_zero_divisors(expression: str) -> None:
    """Quantized-zero constants and branches containing zero refuse before graph emission."""
    neuron = EquationNeuron({"v": expression}, parameters={"tau": 0.0001}, method="map")
    with pytest.raises(ValueError, match="may encode zero"):
        compile_to_ir(neuron)


@pytest.mark.parametrize(
    ("expression", "ranges", "fixed_divisor", "positive"),
    [
        ("I / v if v > 0 else 0", ((1, 32767),), None, False),
        ("I / v if v < 0 else 0", ((-32768, -1),), None, False),
        ("0 if v >= 0 else I / v", ((-32768, -1),), None, False),
        ("I / v if v < 0 or v > 0 else 0", ((-32768, -1), (1, 32767)), None, False),
        ("I / v if 0 < v else 0", ((1, 32767),), None, False),
        ("I / v if 0 < v < 2 else 0", ((1, 511),), None, False),
        ("I / v if v > 0 and I / v > 0 else 0", ((1, 32767),), None, True),
        ("I / tau", ((-32768, 32767),), 768, False),
        ("I / -3", ((-32768, 32767),), -768, False),
    ],
)
def test_native_equation_preserves_admitted_quotients_and_source_guards(
    expression: str,
    ranges: tuple[tuple[int, int], ...],
    fixed_divisor: int | None,
    positive: bool,
    tmp_path: Path,
) -> None:
    """Actual zero/negative/positive fabric traces match independent signed quotient words."""
    neuron = EquationNeuron({"v": "I", "w": expression}, parameters={"tau": 3.0}, method="map")
    result = compile_to_ir(neuron, "division_subject", 16, 8, overflow="wrap")
    sv = result.graph.emit_sv()
    assert " / " in sv
    top, tb, exe = tmp_path / "top.sv", tmp_path / "tb.sv", tmp_path / "sim.vvp"
    top.write_text(sv)
    bench = "module tb;reg clk=0,rst_n=1;reg signed [15:0] I_t=0;wire signed [15:0] v,w;wire spike;division_subject dut(.clk(clk),.rst_n(rst_n),.I_t(I_t),.v_out(v),.w_out(w),.spike_out(spike));initial begin #1;rst_n=0;#1;rst_n=1;"
    previous = 0
    for value in [0, 384, -768, 512, 256, 0, 256]:
        allowed = any(low <= previous <= high for low, high in ranges)
        divisor = previous if fixed_divisor is None else fixed_divisor
        expected = (
            0
            if not allowed
            else (abs(value << 8) // abs(divisor)) * (-1 if (value < 0) != (divisor < 0) else 1)
        )
        if positive and expected <= 0:
            expected = 0
        bench += f"I_t=16'h{value & 65535:x};#2;clk=1;#1;if(w !== 16'h{expected & 65535:x}) $fatal(1,\"quotient mismatch\");clk=0;#1;"
        previous = value
    tb.write_text(bench + '$display("OBS guarded");$finish;end endmodule')
    subprocess.run(
        ["iverilog", "-g2012", "-s", "tb", "-o", str(exe), str(top), str(tb)],
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    observed = subprocess.run(
        ["vvp", str(exe)], capture_output=True, text=True, check=True, timeout=60
    )
    assert "OBS guarded" in observed.stdout

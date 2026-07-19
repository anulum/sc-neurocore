# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Alpha-synapse Python-to-Verilog fidelity contracts

"""Paired-schema and generated Q32.32 state/event-trajectory parity."""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile

if sys.version_info >= (3, 11):
    import tomllib
else:  # Python 3.10
    import tomli as tomllib
from pathlib import Path

import numpy as np

from sc_neurocore.compiler.equation_compiler import Q88
from sc_neurocore.neurons.models.alpha import AlphaNeuron
from sc_neurocore.neurons.universal_dsl import UniversalNeuron
from tests.cosim_support import HAS_IVERILOG

_SCHEMA_DIR = Path(__file__).resolve().parents[1] / "src/sc_neurocore/neurons/model_schemas"
_DATA_WIDTH = 64
_FRACTION = 32
_SCALE = float(1 << _FRACTION)
# Enrolled grid-exact operating point: -dt/tau_v = -0.125, -dt/tau_exc = -0.25,
# -dt/tau_inh = -0.5 all land exactly on the generated exp lookup grid (0.125
# argument spacing), and all three rates stay distinct so the exact alpha
# convolution uses the general branch the schema encodes. The inhibitory drive
# is the overridable schema parameter (single-input RTL contract).
_CONFIG = {"tau_v": 8.0, "tau_exc": 4.0, "tau_inh": 2.0, "dt": 1.0}
_INDEX = np.arange(256, dtype=np.float64)
_EXC = 2.0 * np.sin(_INDEX * 0.173) + 1.2 * np.cos(_INDEX * 0.071) + 0.03125 * np.sin(_INDEX * 0.41)


def _signed_literal(value: int) -> str:
    """Return one syntactically valid signed Q32.32 Verilog literal."""
    if value < 0:
        return f"-{_DATA_WIDTH}'sd{-value}"
    return f"{_DATA_WIDTH}'sd{value}"


def _q3232_rtl_trace(
    exc: np.ndarray,
    inh_drive: float,
) -> list[tuple[float, float, float, float, float, int]]:
    """Return public state and event outputs from generated Q32.32 RTL."""
    neuron = UniversalNeuron.from_schema(
        "alpha",
        parameter_overrides={**_CONFIG, "inh_drive": inh_drive},
    )
    module_name = "sc_alpha_q3232"
    verilog = neuron.to_verilog(
        module_name=module_name,
        data_width=_DATA_WIDTH,
        fraction=_FRACTION,
    )
    quantiser = Q88(data_width=_DATA_WIDTH, fraction=_FRACTION)
    stimuli: list[str] = []
    for exc_value in exc:
        stimuli.extend(
            (
                f"        I_t = {_signed_literal(quantiser.encode(float(exc_value)))};",
                "        @(posedge clk); #1;",
                '        $display("ALPHA_TRACE %0d %0d %0d %0d %0d %0d", v_out, a_exc_out, i_exc_out, a_inh_out, i_inh_out, spike_out);',
            )
        )
    testbench = "\n".join(
        (
            "`timescale 1ns / 1ps",
            f"module tb_{module_name};",
            "reg clk = 1'b0;",
            "reg rst_n = 1'b0;",
            f"reg signed [{_DATA_WIDTH - 1}:0] I_t = 0;",
            "wire spike_out;",
            f"wire signed [{_DATA_WIDTH - 1}:0] v_out;",
            f"wire signed [{_DATA_WIDTH - 1}:0] a_exc_out;",
            f"wire signed [{_DATA_WIDTH - 1}:0] i_exc_out;",
            f"wire signed [{_DATA_WIDTH - 1}:0] a_inh_out;",
            f"wire signed [{_DATA_WIDTH - 1}:0] i_inh_out;",
            "always #5 clk = ~clk;",
            f"{module_name} uut (",
            "    .clk(clk), .rst_n(rst_n), .I_t(I_t),",
            "    .spike_out(spike_out), .a_exc_out(a_exc_out), .i_exc_out(i_exc_out),",
            "    .a_inh_out(a_inh_out), .i_inh_out(i_inh_out), .v_out(v_out)",
            ");",
            "initial begin",
            "    #23; rst_n = 1'b1;",
            *stimuli,
            "    $finish;",
            "end",
            "endmodule",
        )
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        rtl_path = root / f"{module_name}.v"
        testbench_path = root / f"tb_{module_name}.v"
        executable_path = root / f"tb_{module_name}"
        rtl_path.write_text(verilog, encoding="utf-8")
        testbench_path.write_text(testbench, encoding="utf-8")
        subprocess.run(
            [
                "iverilog",
                "-g2012",
                "-o",
                str(executable_path),
                str(rtl_path),
                str(testbench_path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
        simulation = subprocess.run(
            ["vvp", str(executable_path)],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
    rows = re.findall(
        r"^ALPHA_TRACE (-?\d+) (-?\d+) (-?\d+) (-?\d+) (-?\d+) (\d+)$",
        simulation.stdout,
        re.MULTILINE,
    )
    assert len(rows) == len(exc)
    return [
        (
            int(v) / _SCALE,
            int(a_exc) / _SCALE,
            int(i_exc) / _SCALE,
            int(a_inh) / _SCALE,
            int(i_inh) / _SCALE,
            int(event),
        )
        for v, a_exc, i_exc, a_inh, i_inh, event in rows
    ]


def test_required_cosimulation_tool_is_available() -> None:
    """Keep the hardware parity lane fail-closed rather than skipped."""
    assert HAS_IVERILOG


def test_schema_formats_are_identical() -> None:
    """Keep the authored TOML and Studio JSON schemas structurally identical."""
    with (_SCHEMA_DIR / "alpha.toml").open("rb") as handle:
        toml_schema = tomllib.load(handle)
    json_schema = json.loads((_SCHEMA_DIR / "alpha.json").read_text(encoding="utf-8"))
    assert toml_schema == json_schema


def test_schemas_match_the_hand_exact_flow() -> None:
    """Preserve the varied dual-drive trajectory in both schema formats."""
    inh = 0.6 + 0.3 * np.cos(_INDEX * 0.031)
    hand = AlphaNeuron()
    schemas = (
        UniversalNeuron.from_schema(_SCHEMA_DIR / "alpha.toml"),
        UniversalNeuron.from_schema(_SCHEMA_DIR / "alpha.json"),
    )
    for exc_value, inh_value in zip(_EXC, inh):
        expected = hand.step(float(exc_value), float(inh_value))
        for schema in schemas:
            got = schema.step(I=float(exc_value), inh_drive=float(inh_value))
            assert got == expected
            for key in ("v", "a_exc", "i_exc", "a_inh", "i_inh"):
                assert abs(schema.state[key] - getattr(hand, key)) < 5.0e-12


def test_q3232_rtl_tracks_exact_flow_and_complete_event_vector() -> None:
    """Bound the generated fixed-point datapath on the public outputs.

    The enrolled operating point places all exponential-relaxation arguments
    exactly on the 0.125-step lookup grid, so the RTL decay factors equal the
    analytic values up to Q32.32 quantisation. Across this 256-step,
    sign-changing excitatory drive with the inhibitory level enrolled at 0.5,
    the declared 0.01 envelope covers the measured fixed-point error, and
    every emitted event is a candidate crossing with the somatic reset.
    """
    inh_drive = 0.5
    hand = AlphaNeuron(**_CONFIG)
    expected: list[tuple[float, float, float, float, float, int]] = []
    for exc_value in _EXC:
        spike = hand.step(float(exc_value), inh_drive)
        expected.append((hand.v, hand.a_exc, hand.i_exc, hand.a_inh, hand.i_inh, spike))
    actual = _q3232_rtl_trace(_EXC, inh_drive)
    for column, name in ((0, "v"), (1, "a_exc"), (2, "i_exc"), (3, "a_inh"), (4, "i_inh")):
        error = max(
            abs(got[column] - want[column]) for got, want in zip(actual, expected, strict=True)
        )
        assert error < 0.01, f"{name} envelope exceeded: {error}"
    actual_events = [row[5] for row in actual]
    expected_events = [row[5] for row in expected]
    assert actual_events == expected_events


def test_q3232_rtl_preserves_spike_somatic_reset_at_two_inhibitory_levels() -> None:
    """Every RTL spike resets v to v_rest at both enrolled inhibitory levels."""
    exc = np.full(512, 2.0)
    for inh_drive in (0.0, 0.5):
        hand = AlphaNeuron(**_CONFIG)
        hand_trace: list[tuple[float, int]] = []
        for exc_value in exc:
            spike = hand.step(float(exc_value), inh_drive)
            hand_trace.append((hand.v, spike))
        actual = _q3232_rtl_trace(exc, inh_drive)
        hand_spikes = [index for index, (_v, spike) in enumerate(hand_trace) if spike == 1]
        rtl_spikes = [index for index, row in enumerate(actual) if row[5] == 1]
        assert rtl_spikes == hand_spikes
        for index in rtl_spikes:
            assert abs(actual[index][0] - 0.0) < 0.01
            assert hand_trace[index][0] == 0.0

# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Adaptive-threshold Python-to-Verilog fidelity contracts

"""Paired-schema and generated Q32.32 state/event-trajectory parity."""

from __future__ import annotations

import json
import math
import re
import subprocess
import sys
import tempfile

if sys.version_info >= (3, 11):
    import tomllib
else:  # Python 3.10
    import tomli as tomllib
from pathlib import Path

from sc_neurocore.compiler.equation_compiler import Q88
from sc_neurocore.neurons.models.adaptive_threshold_if import AdaptiveThresholdIFNeuron
from sc_neurocore.neurons.universal_dsl import UniversalNeuron
from tests.cosim_support import HAS_IVERILOG

_SCHEMA_DIR = Path(__file__).resolve().parents[1] / "src/sc_neurocore/neurons/model_schemas"
_DATA_WIDTH = 64
_FRACTION = 32
_SCALE = float(1 << _FRACTION)
# Enrolled grid-exact operating point: -dt/tau_m == -dt/tau_theta == -0.125 lands
# exactly on the generated exp lookup grid (0.125 argument spacing), so the RTL
# decay factors equal the analytic exp(-0.125) up to Q32.32 quantisation.
_CONFIG = {"tau_m": 0.8, "tau_theta": 0.8, "dt": 0.1}
_DRIVES = tuple(
    22.0 * math.sin(index * 0.173)
    + 8.0 * math.cos(index * 0.071)
    + 0.03125 * math.sin(index * 0.41)
    for index in range(256)
)


def _signed_literal(value: int) -> str:
    """Return one syntactically valid signed Q32.32 Verilog literal."""
    if value < 0:
        return f"-{_DATA_WIDTH}'sd{-value}"
    return f"{_DATA_WIDTH}'sd{value}"


def _q3232_rtl_trace(drives: tuple[float, ...]) -> list[tuple[float, float, int]]:
    """Return public voltage, threshold, and event outputs from generated Q32.32 RTL."""
    neuron = UniversalNeuron.from_schema("adaptive_threshold_if", parameter_overrides=_CONFIG)
    module_name = "sc_adaptive_threshold_if_q3232"
    verilog = neuron.to_verilog(
        module_name=module_name,
        data_width=_DATA_WIDTH,
        fraction=_FRACTION,
    )
    quantiser = Q88(data_width=_DATA_WIDTH, fraction=_FRACTION)
    stimuli: list[str] = []
    for drive in drives:
        stimuli.extend(
            (
                f"        I_t = {_signed_literal(quantiser.encode(drive))};",
                "        @(posedge clk); #1;",
                '        $display("ADAPTIVE_THRESHOLD_IF_TRACE %0d %0d %0d", v_out, theta_out, spike_out);',
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
            f"wire signed [{_DATA_WIDTH - 1}:0] theta_out;",
            "always #5 clk = ~clk;",
            f"{module_name} uut (",
            "    .clk(clk), .rst_n(rst_n), .I_t(I_t),",
            "    .spike_out(spike_out), .v_out(v_out), .theta_out(theta_out)",
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
        r"^ADAPTIVE_THRESHOLD_IF_TRACE (-?\d+) (-?\d+) (\d+)$",
        simulation.stdout,
        re.MULTILINE,
    )
    assert len(rows) == len(drives)
    return [(int(v) / _SCALE, int(theta) / _SCALE, int(event)) for v, theta, event in rows]


def test_required_cosimulation_tool_is_available() -> None:
    """Keep the hardware parity lane fail-closed rather than skipped."""
    assert HAS_IVERILOG


def test_schema_formats_are_identical() -> None:
    """Keep the authored TOML and Studio JSON schemas structurally identical."""
    with (_SCHEMA_DIR / "adaptive_threshold_if.toml").open("rb") as handle:
        toml_schema = tomllib.load(handle)
    json_schema = json.loads(
        (_SCHEMA_DIR / "adaptive_threshold_if.json").read_text(encoding="utf-8")
    )
    assert toml_schema == json_schema


def test_schemas_match_the_hand_exact_relaxation() -> None:
    """Preserve the varied adaptive-threshold trajectory in both schema formats."""
    hand = AdaptiveThresholdIFNeuron()
    schemas = (
        UniversalNeuron.from_schema(_SCHEMA_DIR / "adaptive_threshold_if.toml"),
        UniversalNeuron.from_schema(_SCHEMA_DIR / "adaptive_threshold_if.json"),
    )
    for drive in _DRIVES:
        expected_v, expected_theta = hand.v, hand.theta
        expected_spike = hand.step(drive)
        for schema in schemas:
            got_spike = schema.step(I=drive)
            assert got_spike == expected_spike
            assert abs(schema.state["v"] - hand.v) < 5.0e-12
            assert abs(schema.state["theta"] - hand.theta) < 5.0e-12
        assert (expected_v, expected_theta) != (hand.v, hand.theta) or expected_spike in (0, 1)


def test_q3232_rtl_tracks_exact_relaxation_and_complete_event_vector() -> None:
    """Bound the generated fixed-point datapath on the public outputs.

    The enrolled operating point places both exponential-relaxation arguments
    exactly on the 0.125-step lookup grid (-dt/tau == -0.125), so the RTL decay
    factors equal the analytic values up to Q32.32 quantisation. Across this
    256-step, sign-changing drive, the declared 0.01 mV envelope covers the
    measured fixed-point error, and every emitted event is a candidate
    crossing with the documented reset and fixed threshold shift.
    """
    hand = AdaptiveThresholdIFNeuron(**_CONFIG)
    expected_v: list[float] = []
    expected_theta: list[float] = []
    expected_events: list[int] = []
    for drive in _DRIVES:
        expected_events.append(hand.step(drive))
        expected_v.append(hand.v)
        expected_theta.append(hand.theta)
    actual = _q3232_rtl_trace(_DRIVES)
    vs = [v for v, _theta, _event in actual]
    thetas = [theta for _v, theta, _event in actual]
    events = [event for _v, _theta, event in actual]
    v_error = max(abs(got - want) for got, want in zip(vs, expected_v, strict=True))
    theta_error = max(abs(got - want) for got, want in zip(thetas, expected_theta, strict=True))
    assert v_error < 0.01, f"v envelope exceeded: {v_error}"
    assert theta_error < 0.01, f"theta envelope exceeded: {theta_error}"
    assert events == expected_events


def test_q3232_rtl_preserves_spike_reset_and_threshold_shift() -> None:
    """Every RTL spike resets v to v_reset and shifts theta by delta_theta."""
    drive = [20.0] * 512
    hand = AdaptiveThresholdIFNeuron(**_CONFIG)
    hand_trace: list[tuple[float, float, int]] = []
    for value in drive:
        spike = hand.step(value)
        hand_trace.append((hand.v, hand.theta, spike))
    actual = _q3232_rtl_trace(tuple(drive))
    hand_spikes = [index for index, (_v, _t, spike) in enumerate(hand_trace) if spike == 1]
    rtl_spikes = [index for index, (_v, _t, event) in enumerate(actual) if event == 1]
    assert rtl_spikes == hand_spikes
    for index in rtl_spikes:
        v_rtl, theta_rtl, _event = actual[index]
        v_hand, theta_hand, _spike = hand_trace[index]
        assert abs(v_rtl - (-65.0)) < 0.01
        assert abs(theta_rtl - theta_hand) < 0.01
        assert v_hand == -65.0

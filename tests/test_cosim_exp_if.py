# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — ExpIF Python-to-Verilog fidelity contracts

"""Source equation, paired-schema, and Q32.32 RTL parity for ExpIF."""

from __future__ import annotations

from pathlib import Path
import hashlib
import subprocess
import tempfile

import pytest
import numpy as np

from sc_neurocore.neurons.models.expif import ExpIFNeuron
from sc_neurocore.neurons.universal_dsl import UniversalNeuron
from tests.cosim_support import HAS_IVERILOG, _python_spike_count, verilog_spike_count_method

_Q3232_GOLDENS = ((0.0, 0), (5.0, 0), (10.0, 1), (20.0, 2), (50.0, 5), (100.0, 9))


def _q3232_complete_trace(n_steps: int, current: float) -> tuple[np.ndarray, np.ndarray]:
    """Run generated compatibility RTL and return every event and voltage row."""
    schema = UniversalNeuron.from_schema("exp_if", method_override="rk4")
    module_name = "sc_expif_complete_q32_32"
    verilog = schema.to_verilog(module_name=module_name, data_width=64, fraction=32)
    current_q = round(current * (1 << 32))
    testbench = f"""
`timescale 1ns/1ps
module tb;
  reg clk = 0;
  reg rst_n = 0;
  reg signed [63:0] I_t = 64'sd{current_q};
  wire spike_out;
  wire signed [63:0] v_out;
  {module_name} uut(.clk(clk), .rst_n(rst_n), .I_t(I_t),
                    .spike_out(spike_out), .v_out(v_out));
  always #5 clk = ~clk;
  integer index;
  initial begin
    #2 rst_n = 0;
    #8 rst_n = 1;
    for (index = 0; index < {n_steps}; index = index + 1) begin
      @(posedge clk); #1;
      $display("ROW %0d %0d %0d", index, spike_out, v_out);
    end
    $finish;
  end
endmodule
"""
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        rtl = root / "model.v"
        tb = root / "tb.v"
        executable = root / "sim"
        rtl.write_text(verilog, encoding="utf-8")
        tb.write_text(testbench, encoding="utf-8")
        compiled = subprocess.run(
            ["iverilog", "-g2012", "-o", str(executable), str(rtl), str(tb)],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        if compiled.returncode:
            raise RuntimeError(compiled.stderr)
        simulated = subprocess.run(
            ["vvp", str(executable)],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
    rows = [line.split() for line in simulated.stdout.splitlines() if line.startswith("ROW ")]
    events = np.asarray([int(row[2]) for row in rows], dtype=np.uint8)
    voltage = np.asarray([int(row[3]) / (1 << 32) for row in rows], dtype=np.float64)
    return voltage, events


def test_expif_schema_formats_match_the_hand_rk4_sequence() -> None:
    """TOML and JSON preserve operation order over a varied driven sequence."""
    schema_dir = Path(__file__).resolve().parents[1] / "src/sc_neurocore/neurons/model_schemas"
    hand = ExpIFNeuron()
    toml_schema = UniversalNeuron.from_schema(schema_dir / "exp_if.toml")
    json_schema = UniversalNeuron.from_schema(schema_dir / "exp_if.json")
    currents = (0.0, 5.0, 10.0, 20.0, 50.0, 100.0, 20.0, 5.0) * 125
    spikes = 0
    max_error = 0.0

    for current in currents:
        hand_spike = hand.step(current)
        spikes += hand_spike
        assert int(bool(toml_schema.step(I=current))) == hand_spike
        assert int(bool(json_schema.step(I=current))) == hand_spike
        max_error = max(
            max_error,
            abs(toml_schema.state["v"] - hand.v),
            abs(json_schema.state["v"] - hand.v),
        )

    assert spikes > 0
    assert max_error <= 2.0e-10


@pytest.mark.skipif(not HAS_IVERILOG, reason="Icarus Verilog not available")
@pytest.mark.parametrize(
    ("current", "expected_spikes"),
    _Q3232_GOLDENS,
    ids=[f"I={current:g}" for current, _expected in _Q3232_GOLDENS],
)
def test_expif_q3232_spike_parity(current: float, expected_spikes: int) -> None:
    """Match hand, schema, and emitted Q32.32 events over 1,000 RK4 steps.

    Q16.16 cannot represent the steep pre-cutoff exponential product without
    losing the event train. Q32.32 retains enough integer and fractional range;
    the six enrolled points span silence, onset, and sustained firing.
    """
    n_steps = 1_000
    hand = ExpIFNeuron()
    hand_spikes = sum(hand.step(current) for _ in range(n_steps))
    schema_spikes = _python_spike_count("exp_if", n_steps, current)
    verilog_spikes = verilog_spike_count_method("exp_if", n_steps, current, 64, 32, "rk4")
    assert hand_spikes == schema_spikes == verilog_spikes == expected_spikes


@pytest.mark.skipif(not HAS_IVERILOG, reason="Icarus Verilog not available")
def test_expif_q3232_complete_event_vector_and_final_state() -> None:
    """Bind every RTL event and final Q32.32 state to an exact golden packet.

    The fixed-point lane preserves the two-event operating point but is not
    falsely promoted to float64 event-time identity: LUT quantisation shifts
    those events to indices 361 and 792. The complete bit-level packet is
    therefore frozen separately from the source and SC-float profiles.
    """
    hand = ExpIFNeuron.sc_rk4_compatibility()
    hand_voltage, _refractory, hand_events = hand.simulate_complete(1_000, 20.0, backend="python")
    rtl_voltage, rtl_events = _q3232_complete_trace(1_000, 20.0)

    assert int(rtl_events.sum()) == int(hand_events.sum()) == 2
    assert np.flatnonzero(rtl_events).tolist() == [361, 792]
    assert hashlib.sha256(rtl_events.tobytes()).hexdigest() == (
        "0c7e93d32a8d7f5d46956bd3928fc01757c8dbefdcb0f2b836a66da3b327c93a"
    )
    voltage_q = np.rint(rtl_voltage * (1 << 32)).astype("<i8")
    assert int(voltage_q[-1]) == -256_410_603_518
    assert hashlib.sha256(voltage_q.tobytes()).hexdigest() == (
        "e25ea92fbac80129c318f48aca0cc7095d3362345379278355eefb12ce5baf1a"
    )
    assert hand_voltage[-1] == pytest.approx(-59.00168087076545, abs=1.0e-12)

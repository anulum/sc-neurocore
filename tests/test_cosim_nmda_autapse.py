# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Wang NMDA-autapse bounded Q16.16 co-simulation

"""Bit-exact integer-oracle and source-envelope checks for NMDA RTL."""

from __future__ import annotations

from pathlib import Path
import re
import subprocess
import tempfile

import numpy as np
import numpy.typing as npt

from sc_neurocore.neurons.models.nmda_neuron import NMDANeuron
from tests.cosim_support import HAS_IVERILOG
from tests.toolchain_support import require_executable

_ROOT = Path(__file__).resolve().parents[1]
_RTL = _ROOT / "hdl/formal/catalogue/sc_nmda_autapse.v"
_SCALE = 65536
_DT = 3277
_HALF_DT = 1638
_ONE = 65536
_V_REST = -4587520
_V_RESET = -3866624
_V_THRESHOLD = -3407872
_V_MIN = -7864320
_V_MAX = 5242880
_REF_PERIOD = 131072
_MG_SAMPLES = (
    137,
    187,
    254,
    346,
    471,
    641,
    871,
    1182,
    1601,
    2163,
    2914,
    3910,
    5218,
    6915,
    9080,
    11786,
    15083,
    18978,
    23411,
    28251,
    33302,
    38326,
    43096,
    47424,
    51196,
    54367,
    56954,
    59014,
    60622,
    61858,
    62798,
    63505,
    64034,
    64428,
    64719,
    64935,
    65094,
    65211,
    65298,
    65361,
    65408,
)
_DRIVE = (1.0, 2.0, 3.0, 0.5, 0.0, 2.0, 0.2, 1.5)


def _q(value: float) -> int:
    return round(value * _SCALE)


def _qmul(left: int, right: int) -> int:
    return (left * right) >> 16


def _mg_block(voltage: int) -> int:
    if voltage <= _V_MIN:
        return _MG_SAMPLES[0]
    if voltage >= _V_MAX:
        return _MG_SAMPLES[-1]
    shifted = voltage - _V_MIN
    index, remainder = divmod(shifted, 5 * _SCALE)
    lower = _MG_SAMPLES[index]
    return lower + (_MG_SAMPLES[index + 1] - lower) * remainder // (5 * _SCALE)


def _derivatives(v: int, x_nmda: int, s_nmda: int, ca: int, current: int) -> tuple[int, ...]:
    leak = _qmul(1638, _V_REST - v)
    nmda = _qmul(6554, _mg_block(v))
    nmda = _qmul(nmda, s_nmda)
    nmda = _qmul(nmda, -v)
    return (
        _qmul(leak + nmda + current, 131072),
        -_qmul(x_nmda, 32768),
        _qmul(x_nmda, _ONE - s_nmda) - _qmul(s_nmda, 819),
        -_qmul(ca, 819),
    )


def _integer_trace(steps: int) -> npt.NDArray[np.int64]:
    v, x_nmda, s_nmda, ca, refractory = _V_REST, 0, 0, 0, 0
    rows: list[tuple[int, ...]] = []
    for index in range(steps):
        current = _q(_DRIVE[index % len(_DRIVE)])
        active = refractory <= 0
        integration_v = v if active else _V_RESET
        k1 = _derivatives(integration_v, x_nmda, s_nmda, ca, current)
        midpoint = (
            integration_v + _qmul(k1[0], _HALF_DT),
            x_nmda + _qmul(k1[1], _HALF_DT),
            s_nmda + _qmul(k1[2], _HALF_DT),
            ca + _qmul(k1[3], _HALF_DT),
        )
        k2 = _derivatives(*midpoint, current)
        candidate = [
            integration_v + _qmul(k2[0], _DT),
            x_nmda + _qmul(k2[1], _DT),
            s_nmda + _qmul(k2[2], _DT),
            ca + _qmul(k2[3], _DT),
        ]
        refractory_next = 0 if refractory <= _DT else refractory - _DT
        event = int(active and candidate[0] >= _V_THRESHOLD)
        if not active:
            candidate[0] = _V_RESET
        elif event:
            candidate[0] = _V_RESET
            candidate[1] += _ONE
            candidate[3] += 13107
            refractory_next = _REF_PERIOD
        v = min(_V_MAX, max(_V_MIN, candidate[0]))
        x_nmda = max(0, candidate[1])
        s_nmda = min(_ONE, max(0, candidate[2]))
        ca = max(0, candidate[3])
        refractory = refractory_next
        rows.append((v, x_nmda, s_nmda, ca, refractory, event))
    return np.asarray(rows, dtype=np.int64)


def _literal(value: float) -> str:
    encoded = _q(value)
    return f"-32'sd{-encoded}" if encoded < 0 else f"32'sd{encoded}"


def _rtl_trace(steps: int) -> npt.NDArray[np.int64]:
    drives: list[str] = []
    for index in range(steps):
        drives.extend(
            (
                f"current={_literal(_DRIVE[index % len(_DRIVE)])};",
                "@(posedge clk); #1;",
                '$display("NMDA_TRACE %0d %0d %0d %0d %0d %0d", v, x, s, ca, refractory, event_out);',
            )
        )
    testbench = "\n".join(
        (
            "module tb;",
            "reg clk=0; reg rst_n=0; reg signed [31:0] current=0;",
            "wire signed [31:0] v, x, s, ca, refractory; wire event_out;",
            "always #5 clk=~clk;",
            "sc_nmda_autapse uut(.clk(clk),.rst_n(rst_n),.current_t(current),",
            ".v_out(v),.x_nmda_out(x),.s_nmda_out(s),.ca_out(ca),",
            ".refractory_out(refractory),.event_out(event_out));",
            "initial begin #23; rst_n=1;",
            *drives,
            "$finish; end endmodule",
        )
    )
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        tb = root / "tb.v"
        binary = root / "tb"
        tb.write_text(testbench, encoding="utf-8")
        subprocess.run(
            ["iverilog", "-g2012", "-o", str(binary), str(_RTL), str(tb)],
            check=True,
            capture_output=True,
            text=True,
        )
        output = subprocess.run(
            ["vvp", str(binary)], check=True, capture_output=True, text=True
        ).stdout
    rows = re.findall(
        r"^NMDA_TRACE (-?\d+) (-?\d+) (-?\d+) (-?\d+) (-?\d+) ([01])$",
        output,
        re.MULTILINE,
    )
    assert len(rows) == steps
    return np.asarray([[int(value) for value in row] for row in rows], dtype=np.int64)


def test_required_cosimulation_tool_is_available() -> None:
    assert HAS_IVERILOG


def test_yosys_synthesises_committed_rtl() -> None:
    completed = subprocess.run(
        [
            require_executable("yosys"),
            "-q",
            "-p",
            f"read_verilog {_RTL}; synth -top sc_nmda_autapse -run begin:coarse; check; stat",
        ],
        cwd=_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert completed.returncode == 0, completed.stderr


def test_q1616_rtl_is_bit_exact_to_independent_integer_oracle() -> None:
    np.testing.assert_array_equal(_rtl_trace(512), _integer_trace(512))


def test_q1616_rtl_preserves_source_event_vector_and_state_envelope() -> None:
    neuron = NMDANeuron()
    expected: list[tuple[float, ...]] = []
    for index in range(512):
        event = neuron.step(_DRIVE[index % len(_DRIVE)])
        expected.append(
            (
                neuron.v,
                neuron.x_nmda,
                neuron.s_nmda,
                neuron.ca,
                neuron.refractory_remaining,
                event,
            )
        )
    actual = _rtl_trace(512)
    expected_array = np.asarray(expected)
    np.testing.assert_array_equal(actual[:, 5], expected_array[:, 5])
    errors = np.max(np.abs(actual[:, :5] / _SCALE - expected_array[:, :5]), axis=0)
    assert errors[0] < 0.012
    assert errors[1] < 0.0004
    assert errors[2] < 0.0006
    assert errors[3] < 0.0024
    assert errors[4] < 0.00013

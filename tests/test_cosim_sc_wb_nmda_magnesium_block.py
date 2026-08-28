# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — retained SC NMDA bounded Q32.32 co-simulation

"""Bit-exact FSM oracle and bounded binary64 checks for retained SC NMDA."""

from __future__ import annotations

import math
from pathlib import Path
import re
import subprocess
import tempfile

import numpy as np
import numpy.typing as npt

from sc_neurocore.neurons.models import SCWBNMDAMagnesiumBlockNeuron
from tests.cosim_support import HAS_IVERILOG
from tests.toolchain_support import require_executable

_ROOT = Path(__file__).resolve().parents[1]
_RTL = _ROOT / "hdl/formal/catalogue/sc_wb_nmda_magnesium_block.v"
_SCALE = 1 << 32
_ONE = _SCALE
_V_REST = -65 * _SCALE
_V_MIN = -100 * _SCALE
_V_MAX = 60 * _SCALE
_V_THRESHOLD = -20 * _SCALE
_SUB_DT = round(0.01 * _SCALE)


def _q(value: float) -> int:
    return round(value * _SCALE)


def _qmul(left: int, right: int) -> int:
    return (left * right) >> 32


def _trunc_div(numerator: int, denominator: int) -> int:
    quotient = abs(numerator) // abs(denominator)
    return -quotient if (numerator < 0) != (denominator < 0) else quotient


def _exact_rates(voltage: float) -> tuple[float, ...]:
    d_m = voltage + 35.0
    alpha_m = 1.0 if abs(d_m) < 1.0e-7 else 0.1 * d_m / (1.0 - math.exp(-d_m / 10.0))
    beta_m = 4.0 * math.exp(-(voltage + 60.0) / 18.0)
    d_n = voltage + 34.0
    alpha_n = 0.1 if abs(d_n) < 1.0e-7 else 0.01 * d_n / (1.0 - math.exp(-d_n / 10.0))
    return (
        alpha_m / (alpha_m + beta_m),
        0.07 * math.exp(-(voltage + 58.0) / 20.0),
        1.0 / (1.0 + math.exp(-(voltage + 28.0) / 10.0)),
        alpha_n,
        0.125 * math.exp(-(voltage + 44.0) / 80.0),
        1.0 / (1.0 + math.exp(-0.062 * voltage) / 3.57),
    )


_RATE_SAMPLES = tuple(
    tuple(_q(value) for value in _exact_rates(float(voltage))) for voltage in range(-100, 61, 5)
)
_DRIVE_SAMPLES = tuple(
    _q(current / (current + 5.0)) if current else 0
    for current in (index * 0.5 for index in range(41))
)


def _drive(current: int) -> int:
    if current <= 0:
        return 0
    if current >= _q(20.0):
        return _DRIVE_SAMPLES[-1]
    index, remainder = divmod(current, _q(0.5))
    lower = _DRIVE_SAMPLES[index]
    return lower + _trunc_div((_DRIVE_SAMPLES[index + 1] - lower) * remainder, _q(0.5))


def _rates(voltage: int) -> tuple[int, ...]:
    if voltage <= _V_MIN:
        return _RATE_SAMPLES[0]
    if voltage >= _V_MAX:
        return _RATE_SAMPLES[-1]
    shifted = voltage - _V_MIN
    index, remainder = divmod(shifted, 5 * _SCALE)
    return tuple(
        lower + _trunc_div((upper - lower) * remainder, 5 * _SCALE)
        for lower, upper in zip(_RATE_SAMPLES[index], _RATE_SAMPLES[index + 1])
    )


def _integer_trace(current: float, steps: int) -> npt.NDArray[np.int64]:
    v, h, n, s_nmda = _V_REST, _q(0.6), _q(0.32), 0
    encoded_current = _q(current)
    rows: list[tuple[int, ...]] = []
    for _ in range(steps):
        drive = _drive(encoded_current)
        gate_step = _q(0.05 if drive > s_nmda else 0.005)
        s_nmda = min(_ONE, max(0, s_nmda + _qmul(gate_step, drive - s_nmda)))
        event = 0
        for _ in range(50):
            m_inf, alpha_h, beta_h, alpha_n, beta_n, mg_block = _rates(v)
            dh = _qmul(_q(5.0), _qmul(alpha_h, _ONE - h) - _qmul(beta_h, h))
            dn = _qmul(_q(5.0), _qmul(alpha_n, _ONE - n) - _qmul(beta_n, n))
            h += _qmul(_SUB_DT, dh)
            n += _qmul(_SUB_DT, dn)
            m_cubed = _qmul(_qmul(m_inf, m_inf), m_inf)
            n_squared = _qmul(n, n)
            n_fourth = _qmul(n_squared, n_squared)
            i_na = _qmul(_qmul(_q(35.0), _qmul(m_cubed, h)), v - _q(55.0))
            i_k = _qmul(_qmul(_q(9.0), n_fourth), v - _q(-90.0))
            i_nmda = _qmul(_qmul(_qmul(_q(0.5), s_nmda), mg_block), v)
            i_l = _qmul(_q(0.1), v - _V_REST)
            v += _qmul(_SUB_DT, -i_na - i_k - i_nmda - i_l + encoded_current)
            if v >= _V_THRESHOLD:
                event = 1
                v = _V_REST
        v = min(_V_MAX, max(_V_MIN, v))
        h = min(_ONE, max(0, h))
        n = min(_ONE, max(0, n))
        rows.append((v, h, n, s_nmda, event))
    return np.asarray(rows, dtype=np.int64)


def _literal(value: float) -> str:
    encoded = _q(value)
    return f"-64'sd{-encoded}" if encoded < 0 else f"64'sd{encoded}"


def _rtl_trace(current: float, steps: int) -> npt.NDArray[np.int64]:
    drives: list[str] = []
    for _ in range(steps):
        drives.extend(
            (
                f"current={_literal(current)}; start=1;",
                "@(posedge clk); #1; start=0; wait(ready); #1;",
                '$display("SC_NMDA_TRACE %0d %0d %0d %0d %0d", v, h, n, s, event_out);',
                "@(posedge clk); #1;",
            )
        )
    testbench = "\n".join(
        (
            "module tb;",
            "reg clk=0; reg rst_n=0; reg start=0; reg signed [63:0] current=0;",
            "wire signed [63:0] v,h,n,s; wire event_out,ready,busy; always #5 clk=~clk;",
            "sc_wb_nmda_magnesium_block uut(.clk(clk),.rst_n(rst_n),.start(start),",
            ".current_t(current),.v_out(v),.h_out(h),.n_out(n),.s_nmda_out(s),",
            ".event_out(event_out),.ready(ready),.busy(busy));",
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
        r"^SC_NMDA_TRACE (-?\d+) (-?\d+) (-?\d+) (-?\d+) ([01])$",
        output,
        re.MULTILINE,
    )
    assert len(rows) == steps
    return np.asarray([[int(value) for value in row] for row in rows], dtype=np.int64)


def _source_trace(current: float, steps: int) -> npt.NDArray[np.float64]:
    neuron = SCWBNMDAMagnesiumBlockNeuron()
    rows = []
    for _ in range(steps):
        event = neuron.step(current)
        rows.append((neuron.v, neuron.h, neuron.n, neuron.s_nmda, event))
    return np.asarray(rows)


def test_required_cosimulation_tool_is_available() -> None:
    assert HAS_IVERILOG


def test_yosys_synthesises_committed_rtl() -> None:
    completed = subprocess.run(
        [
            require_executable("yosys"),
            "-q",
            "-p",
            f"read_verilog {_RTL}; synth -top sc_wb_nmda_magnesium_block -run begin:coarse; check; stat",
        ],
        cwd=_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert completed.returncode == 0, completed.stderr


def test_q3232_fsm_is_bit_exact_to_integer_oracle() -> None:
    np.testing.assert_array_equal(_rtl_trace(0.0, 64), _integer_trace(0.0, 64))
    np.testing.assert_array_equal(_rtl_trace(5.0, 32), _integer_trace(5.0, 32))


def test_q3232_fsm_preserves_bounded_source_vectors() -> None:
    for current, steps, bounds in (
        (0.0, 64, (0.08, 0.004, 0.0012, 1.0e-12)),
        (5.0, 32, (4.6, 0.037, 0.016, 2.0e-9)),
    ):
        actual = _rtl_trace(current, steps)
        expected = _source_trace(current, steps)
        np.testing.assert_array_equal(actual[:, 4], expected[:, 4])
        errors = np.max(np.abs(actual[:, :4] / _SCALE - expected[:, :4]), axis=0)
        assert np.all(errors < np.asarray(bounds))

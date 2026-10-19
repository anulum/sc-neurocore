# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — numeric diagnostics in exact arithmetic, matching the selected RTL rounding

"""Diagnostics of huge finite values and composite divisors are exact and complete.

RIGOR reviews 285 and 289 found three diagnostic gaps:

* NEURO-Q88-PRECISION-FINITE-01: ``precision_report`` and the ``precision`` CLI
  raised a host ``OverflowError`` for ``dt`` or a parameter of ``1e308``; both now
  report the exact code and the range problem, and the CLI exits normally.
* NEURO-NUMERIC-COMPOSITE-FINITE-01: the contract folded ``p ** 2`` in host floats
  and raised for ``p = 1e308``; real values now fold as fractions and the
  out-of-range parameter is reported as blocking.
* NEURO-NUMERIC-BANKERS-DIVISOR-01: the divisor check treated bankers' rounding as
  truncation and refused ``c / (p * p)`` whose RTL divisor is 1; it now rounds as
  the RTL does, half to even, so an exact tie that rounds to zero is still refused.

A dt outside the format, which the RTL and the C/Rust kernels multiplied by in
its wrapped form (1.0 at Q1.15 as -1.0), is refused by every emitter wherever dt
is multiplied (not by a map). Each case has a healthy control.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, cast

import pytest

from sc_neurocore.compiler.hardware_numeric_contract import hardware_numeric_contract
from sc_neurocore.compiler.intelligence.bit_true_kernel import generate_bittrue_kernel_from_neuron
from sc_neurocore.compiler.q_format import QFormat
from sc_neurocore.compiler.verilog_compiler import compile_to_datapath, compile_to_verilog
from sc_neurocore.compiler.verilog_compiler_config import Q88
from sc_neurocore.neurons.equation_builder import EquationNeuron
from tests.bit_true_cosim_support import HAS_COSIM, _verilog_trace

_REPO = Path(__file__).resolve().parents[1]


def test_the_precision_report_states_a_huge_dt_and_parameter() -> None:
    """A dt of 1e308 is OUT OF RANGE; p = 1e308 carries the overflow warning; 0.5 is exact."""
    huge_dt = Q88().precision_report(1e308, {"p": 0.5})
    assert "OUT OF RANGE" in huge_dt and "p=0.5 → Q-value=128 (error=0.0%)" in huge_dt
    huge_p = Q88().precision_report(0.5, {"p": 1e308})
    assert "⚠ Overflow: p=1e+308 exceeds Q8.8" in huge_p
    assert "dt=0.5 → Q-value=128 (actual=0.500000, error=0.0%) ✓" in huge_p


@pytest.mark.parametrize(
    ("dt", "parameter"), [(1e308, 0.5), (0.5, 1e308), (0.5, 0.5)], ids=["dt", "param", "control"]
)
def test_the_precision_cli_completes(dt: float, parameter: float) -> None:
    """The real CLI on a schema file exits 0 without a traceback; huge values fit no mode."""
    schema = {
        "metadata": {"schema_version": 1, "name": "finite precision"},
        "state": {"v": 0.0},
        "parameters": {"p": parameter},
        "dynamics": {"v": "0.0"},
        "integration": {"dt": dt, "method": "map"},
    }
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "schema.json"
        path.write_text(json.dumps(schema))
        done = subprocess.run(
            [sys.executable, "-B", "-m", "sc_neurocore.neurons.dsl_cli", "precision", str(path)],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=_REPO,
            env={"PYTHONPATH": str(_REPO / "src"), "PATH": ""},
        )
    assert done.returncode == 0 and "Traceback" not in done.stderr
    huge = max(dt, parameter) > 1.0
    assert ("Compatible modes: NONE" in done.stdout) is huge


@pytest.mark.parametrize(
    ("parameter", "expression", "representable"),
    [(1e308, "v / p", False), (1e308, "v / (p ** 2)", False), (2.0, "v / (p ** 2)", True)],
)
def test_a_huge_composite_divisor_is_reported(
    parameter: float, expression: str, representable: bool
) -> None:
    """The contract completes; the out-of-range parameter is the blocking diagnostic."""
    neuron = EquationNeuron(
        equations={"v": expression}, state={"v": 0.5}, parameters={"p": parameter}, dt=0.5
    )
    contract = hardware_numeric_contract(neuron, QFormat.from_string("Q8.8"))
    assert contract.representable is representable
    quantities = cast(list[dict[str, Any]], contract.to_public_dict()["quantities"])
    blocking = [q for q in quantities if q["blocking"]]
    assert [q["name"] for q in blocking] == ([] if representable else ["p"])


def _divisor_neuron(p: float, q: float) -> EquationNeuron:
    """Return the map neuron v = p*q, u = c/(p*q) at Q8.8 (c = 1/256)."""
    return EquationNeuron(
        equations={"v": "p * q", "u": "c / (p * q)"},
        state={"v": 0.0, "u": 0.0},
        parameters={"p": p, "q": q, "c": 1 / 256},
        dt=1.0,
        method="map",
    )


# (p, q) codes at Q8.8 and, per rounding, whether the RTL product p*q is zero.
_PRODUCTS = {
    "12x12": ((12 / 256, 12 / 256), {"truncate": True, "nearest": False, "bankers": False}),
    "16x8-tie": ((16 / 256, 8 / 256), {"truncate": True, "nearest": False, "bankers": True}),
    "16x16": ((16 / 256, 16 / 256), {"truncate": False, "nearest": False, "bankers": False}),
}


@pytest.mark.parametrize("rounding", ["truncate", "nearest", "bankers"])
@pytest.mark.parametrize("case", sorted(_PRODUCTS))
def test_the_divisor_check_rounds_as_the_rtl(case: str, rounding: str) -> None:
    """A divisor is refused exactly when its RTL word is zero under the selected rounding."""
    (p, q), zero = _PRODUCTS[case]
    contract = hardware_numeric_contract(
        _divisor_neuron(p, q), QFormat.from_string("Q8.8"), rounding=rounding
    )
    assert contract.representable is not zero[rounding]


@pytest.mark.skipif(not HAS_COSIM, reason="Icarus Verilog / gcc not available")
@pytest.mark.parametrize("rounding", ["truncate", "nearest", "bankers"])
@pytest.mark.parametrize("case", sorted(_PRODUCTS))
def test_the_rtl_product_is_what_the_check_assumed(case: str, rounding: str) -> None:
    """Icarus: the RTL's p*q code is zero exactly where the contract refuses the divisor."""
    (p, q), zero = _PRODUCTS[case]
    neuron = EquationNeuron(
        equations={"v": "p * q"},
        state={"v": 0.0},
        parameters={"p": p, "q": q},
        dt=1.0,
        method="map",
    )
    with tempfile.TemporaryDirectory() as d:
        rows = _verilog_trace(neuron, "m", 0.0, 2, 16, 8, Path(d), rounding=rounding)
    assert (int(rows[0][1]) == 0) is zero[rounding]


@pytest.mark.parametrize("emit", ["registered", "folded", "c", "rust"])
@pytest.mark.parametrize(
    ("dt", "width", "fraction"), [(1e308, 16, 8), (1.0, 16, 15)], ids=["huge", "one-at-Q1.15"]
)
def test_an_out_of_range_dt_is_refused_by_every_emitter(
    emit: str, dt: float, width: int, fraction: int
) -> None:
    """A dt the word cannot hold wraps (1.0 at Q1.15 was multiplied as -1.0): refused.

    An Euler step multiplies by dt, so every emitter refuses it; a map never
    multiplies by dt and still compiles, as does an Euler step with dt = 0.5.
    """

    def compile_with(step: float, method: str) -> str:
        neuron = EquationNeuron(
            equations={"v": "0.5 - v" if method == "euler" else "0.5"},
            state={"v": 0.0},
            method=method,
            dt=step,
        )
        if emit == "registered":
            return compile_to_verilog(neuron, "m", width, fraction)
        if emit == "folded":
            return compile_to_datapath(neuron, module_name="m", data_width=width, fraction=fraction)
        return generate_bittrue_kernel_from_neuron(
            neuron, "m", data_width=width, fraction=fraction, language=emit
        )

    with pytest.raises(ValueError, match=re.escape(f"Overflow: dt={dt} exceeds Q")):
        compile_with(dt, "euler")
    assert compile_with(dt, "map")
    assert compile_with(0.5, "euler")

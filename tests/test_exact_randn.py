# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — randn() with random_precision="exact": 0.5 LSB + 2**-45 everywhere

"""``random_precision="exact"`` lowers ``randn()`` with 60-bit internals.

RIGOR NEURO-8AE-NORMAL-01 asked for half an LSB at high fractions; the standard
unit (30-bit internals) is within 0.5 LSB + ``2**-24``, many LSB at Q3.29. The
exact mode, chosen by the owner as an option beside the standard one, rounds once
from a transform within ``2**-45`` of the exact value. These cases measure that
bound, check that the integer program equals the reference, that the RTL, the C
kernel (UBSan) and the Rust kernel (overflow checks) equal both, and that the
standard mode is unchanged. Toolchain cases skip without Icarus, gcc or rustc.
"""

from __future__ import annotations

import math
import subprocess
import tempfile
from decimal import Decimal, localcontext
from pathlib import Path

import pytest

from sc_neurocore.compiler import exact_normal as xn
from sc_neurocore.compiler import random_units as ru
from sc_neurocore.compiler.hardware_numeric_contract import hardware_numeric_contract
from sc_neurocore.compiler.intelligence.bit_true_kernel import generate_bittrue_kernel_from_neuron
from sc_neurocore.compiler.q_format import QFormat
from sc_neurocore.compiler.random_unit_programs import program
from sc_neurocore.compiler.verilog_compiler import compile_to_verilog
from sc_neurocore.compiler.verilog_compiler_config import Q88
from sc_neurocore.neurons.equation_builder import EquationNeuron
from tests.bit_true_cosim_support import HAS_COSIM, HAS_RUST, _c_main, _rows, _rust_main, _testbench


def _words() -> list[tuple[int, int]]:
    """Word pairs: every word0 within 1500 of 2**32, the series boundary, extremes, draws."""
    turns = (0, 7, 1 << 30, (1 << 30) + 12345, 1 << 31, (3 << 30) + 99, (1 << 32) - 1)
    near = [(1 << 32) - 1 - d for d in range(1500)]
    edges = [(1 << 32) - 1 - d for d in (xn.NEAR_ONE_WORDS - 1, xn.NEAR_ONE_WORDS)] + [0, 1 << 31]
    pairs = [(w0, w1) for w0 in near + edges for w1 in turns]
    return pairs + [ru.draw(1, step, 0, 2026)[:2] for step in range(4000)]


def test_the_transform_is_within_its_stated_error() -> None:
    """Before rounding, the 60-bit transform is within 2**-45 of the exact Box-Muller value."""
    worst = 0.0
    with localcontext() as ctx:
        ctx.prec = 40
        for w0, w1 in _words():
            radius = xn.sqrt_fraction58(*xn.neg_log_u(w0))
            cos = xn.turn_cosine(w1)
            raw = Decimal(xn.mul60(radius, abs(cos)) * (1 if cos >= 0 else -1)) / (1 << 58)
            exact = (-2 * (Decimal(w0 + 1) / (1 << 32)).ln()).sqrt()
            exact *= Decimal(math.cos(2 * math.pi * w1 / 2**32))
            worst = max(worst, abs(float(raw - exact)))
    assert 0.0 < worst <= xn.EXACT_NORMAL_TRANSFORM_ERROR


@pytest.mark.parametrize(("data_width", "fraction"), [(32, 29), (32, 16), (62, 40), (16, 8)])
def test_every_code_is_within_half_an_lsb_plus_the_bound(data_width: int, fraction: int) -> None:
    """|code - value * 2**fraction| <= 0.5 + 2**(fraction - 45) of the model clipped to the word.

    At Q3.29 the standard unit is up to 32.5 LSB away; the exact mode stays under 0.5005.
    """
    scale = 1 << fraction
    limit = 1 << (data_width - 1)
    bound = 0.5 + xn.EXACT_NORMAL_TRANSFORM_ERROR * scale + 1e-9
    for w0, w1 in _words():
        model = max(-limit / scale, min((limit - 1) / scale, ru.normal_value(w0, w1)))
        assert abs(xn.normal_code_exact(w0, w1, data_width, fraction) - model * scale) <= bound


@pytest.mark.parametrize(
    ("site", "seed", "data_width", "fraction"), [(1, 2026, 32, 29), (3, 7, 16, 8)]
)
def test_the_program_equals_the_reference(
    site: int, seed: int, data_width: int, fraction: int
) -> None:
    """The 64-bit program (every value checked to fit) returns the reference code at every step."""
    unit = program("randn_exact", site, data_width, fraction)
    for step in range(1500):
        w0, w1 = ru.draw(site, step, 0, seed)[:2]
        got = unit.evaluate(0, {"step": step, "seed": seed, "stream": 0})
        assert got == xn.normal_code_exact(w0, w1, data_width, fraction)


def test_an_unknown_precision_is_refused() -> None:
    """An unknown random precision is refused when the format is built."""
    with pytest.raises(ValueError, match="random_precision"):
        Q88(data_width=16, fraction=8, random_precision="extra")


def _neuron() -> EquationNeuron:
    """Return the map neuron v = randn()."""
    return EquationNeuron(
        equations={"v": "randn()"}, state={"v": 0.0}, dt=1.0, method="map", rng_seed=2026
    )


def test_the_standard_mode_is_unchanged_and_the_contract_names_the_mode() -> None:
    """Default emission equals explicit standard; the contract states the exact mode's bound."""
    neuron = _neuron()
    default = compile_to_verilog(neuron, "m", 32, 29)
    assert default == compile_to_verilog(neuron, "m", 32, 29, random_precision="standard")
    assert default != compile_to_verilog(neuron, "m", 32, 29, random_precision="exact")
    contract = hardware_numeric_contract(neuron, QFormat(3, 29), random_precision="exact")
    assert contract.to_public_dict()["random_precision"] == "exact"
    assert contract.arithmetic is not None
    assert "2**-45" in str(contract.arithmetic["randomness"])


def _run(cmd: list[str]) -> str:
    """Run a toolchain command and return its stdout."""
    return subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=300).stdout


@pytest.mark.skipif(not HAS_COSIM, reason="Icarus Verilog / gcc not available")
@pytest.mark.parametrize(("data_width", "fraction"), [(32, 29), (32, 16)])
def test_rtl_c_and_rust_draw_the_reference_codes(data_width: int, fraction: int) -> None:
    """Twenty steps of v = randn(): RTL = C (UBSan) = Rust (overflow checks) = reference."""
    neuron = _neuron()
    steps = 20
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        rtl_source = compile_to_verilog(neuron, "m", data_width, fraction, random_precision="exact")
        (tmp / "m.v").write_text(rtl_source)
        (tmp / "tb.v").write_text(_testbench(neuron, "m", 0.0, steps, data_width, fraction))
        _run(["iverilog", "-g2012", "-o", str(tmp / "s"), str(tmp / "m.v"), str(tmp / "tb.v")])
        rtl = _rows(_run(["vvp", str(tmp / "s")]))
        kernel = generate_bittrue_kernel_from_neuron(
            neuron, "m", data_width=data_width, fraction=fraction, random_precision="exact"
        )
        (tmp / "m.c").write_text(kernel + "\n" + _c_main(neuron, "m", 0, steps, data_width))
        _run(
            [
                "gcc",
                "-O2",
                "-fsanitize=undefined",
                "-fno-sanitize-recover=all",
                "-o",
                str(tmp / "k"),
                str(tmp / "m.c"),
            ]
        )
        c = _rows(_run([str(tmp / "k")]))
        rust = None
        if HAS_RUST:
            source = generate_bittrue_kernel_from_neuron(
                neuron,
                "m",
                data_width=data_width,
                fraction=fraction,
                language="rust",
                random_precision="exact",
            )
            (tmp / "m.rs").write_text(
                "#![allow(non_snake_case, dead_code, unused)]\n"
                + source
                + "\n"
                + _rust_main(neuron, "m", 0, steps, data_width).replace(
                    "spike_out: 0", "spike_out: 0, rng_step: 0", 1
                )
            )
            _run(["rustc", "-C", "overflow-checks=on", "-o", str(tmp / "r"), str(tmp / "m.rs")])
            rust = _rows(_run([str(tmp / "r")]))
    reference = [
        str(xn.normal_code_exact(*ru.draw(0, step, 0, 2026)[:2], data_width, fraction))
        for step in range(steps)
    ]
    assert rtl == c
    assert [row[1] for row in rtl] == reference
    if rust is not None:
        assert rust == c

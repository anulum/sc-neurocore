# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — the diffusion noise xi is the counter-based normal draw over sqrt(dt), on every target

"""``xi`` is ``randn(XI_SITE) / sqrt(dt)`` in the model and compiles bit for bit.

Before 2026-10-19 ``xi`` was drawn from NumPy's process-global generator (not
replayable, no RTL or kernel lowering) and, although the code comment and the
compiler's refusal said ``N/sqrt(dt)``, it computed ``sqrt(dt)·N/sqrt(dt) = N``:
an Euler step added ``N·dt``, whose variance ``dt**2`` is not white noise. By the
owner's decision ``xi`` is now the Philox normal draw of the reserved call site
``XI_SITE`` (sub-step ``k`` reads ``XI_SITE - k``) times ``1/sqrt(dt)``; every
occurrence in a step shares the draw, including Runge–Kutta stages. These cases
check the model's value and variance, the shared draw, the reserved sites, the
legacy ``noise_rng`` path, and that the RTL, the C kernel and the Rust kernel
equal each other and the integer reference.
"""

from __future__ import annotations

import math
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest

from sc_neurocore.compiler import exact_normal as xn
from sc_neurocore.compiler import random_units as ru
from sc_neurocore.compiler.verilog_compiler import compile_to_verilog
from sc_neurocore.neurons.equation_builder import EquationNeuron, from_equations
from sc_neurocore.neurons.equation_random import RESERVED_SITES, XI_SITE
from sc_neurocore.neurons.equation_refusals import EquationRefusal
from sc_neurocore.compiler.intelligence.bit_true_kernel import generate_bittrue_kernel_from_neuron
from tests.bit_true_cosim_support import (
    HAS_COSIM,
    HAS_RUST,
    _c_trace,
    _rows,
    _rust_main,
    _testbench,
    _verilog_trace,
)

_SEED = 2026


def _draw(step: int, site: int = XI_SITE) -> float:
    """Return the model's standard-normal draw of ``site`` at ``step`` (stream 0)."""
    words = ru.draw(site, step, 0, _SEED)
    return ru.normal_value(words[0], words[1])


def test_xi_is_the_site_draw_over_sqrt_dt() -> None:
    """An Euler step of dv/dt = xi adds randn(XI_SITE) · (1/sqrt(dt)) · dt."""
    neuron = from_equations("dv/dt = xi", init={"v": 0.0}, dt=0.25, rng_seed=_SEED)
    for step in range(5):
        before = neuron.state["v"]
        neuron.step(0.0)
        assert neuron.state["v"] - before == pytest.approx(_draw(step) * 2.0 * 0.25, abs=1e-15)


def test_an_euler_increment_has_variance_dt() -> None:
    """Over 4000 steps the increment's variance is dt (white noise), not dt**2 as before."""
    for dt in (0.25, 0.01):
        neuron = from_equations("dv/dt = xi", init={"v": 0.0}, dt=dt, rng_seed=_SEED)
        increments = []
        for _ in range(4000):
            before = neuron.state["v"]
            neuron.step(0.0)
            increments.append(neuron.state["v"] - before)
        assert np.var(increments) == pytest.approx(dt, rel=0.08)


def test_every_occurrence_and_stage_shares_the_draw() -> None:
    """The term xi - xi is zero in a step, and an RK4 step of dv/dt = xi adds xi·dt."""
    zero = from_equations("dv/dt = xi - xi", init={"v": 0.0}, dt=0.5, rng_seed=_SEED)
    rk4 = from_equations("dv/dt = xi", init={"v": 0.0}, dt=0.5, rng_seed=_SEED, method="rk4")
    for step in range(4):
        zero.step(0.0)
        before = rk4.state["v"]
        rk4.step(0.0)
        assert zero.state["v"] == 0.0
        assert rk4.state["v"] - before == pytest.approx(
            _draw(step) / math.sqrt(0.5) * 0.5, abs=1e-15
        )


def test_each_sub_step_reads_its_own_site() -> None:
    """With substeps = 3, sub-step k adds the draw of XI_SITE - k."""
    neuron = EquationNeuron(
        equations={"v": "xi"}, state={"v": 0.0}, dt=1.0, rng_seed=_SEED, substeps=3
    )
    neuron.step(0.0)
    expected = sum(_draw(0, XI_SITE - k) for k in range(3))
    assert neuron.state["v"] == pytest.approx(expected, abs=1e-12)


def test_the_reserved_sites_are_refused_as_explicit_sites() -> None:
    """randn(k) with k at or above 2**31 is refused; below it is accepted."""
    with pytest.raises(EquationRefusal, match="reserved for the diffusion noise"):
        from_equations(f"dv/dt = randn({RESERVED_SITES})", init={"v": 0.0}, dt=1.0)
    assert from_equations(f"dv/dt = randn({RESERVED_SITES - 1})", init={"v": 0.0}, dt=1.0)


def test_a_noise_rng_draws_xi_in_python_and_is_not_compiled() -> None:
    """The legacy generator path draws from it (scaled by 1/sqrt(dt)); the compiler refuses it."""
    rng = np.random.default_rng(5)
    expected = np.random.default_rng(5).standard_normal() / math.sqrt(0.25) * 0.25
    neuron = from_equations("dv/dt = xi", init={"v": 0.0}, dt=0.25, noise_rng=rng)
    neuron.step(0.0)
    assert neuron.state["v"] == pytest.approx(expected, abs=1e-15)
    assert neuron.xi_from_noise_rng and not neuron.uses_random_draws
    with pytest.raises(ValueError, match="noise_rng"):
        compile_to_verilog(neuron, "m", 16, 8)


def test_sub_stepped_xi_is_refused_by_the_compiler() -> None:
    """The sub-stepped datapath does not number one site per sub-step."""
    neuron = EquationNeuron(
        equations={"v": "xi"}, state={"v": 0.0}, dt=1.0, method="map", substeps=2
    )
    with pytest.raises(ValueError, match="substeps"):
        compile_to_verilog(neuron, "m", 16, 8)


def _map_neuron() -> EquationNeuron:
    """Return the map v = xi at dt = 0.25 (xi = 2·randn(XI_SITE))."""
    return EquationNeuron(
        equations={"v": "xi"}, state={"v": 0.0}, dt=0.25, method="map", rng_seed=_SEED
    )


def _traces(neuron_factory: object, steps: int, width: int, fraction: int) -> list[list[list[str]]]:
    """Return the RTL, C and (when rustc exists) Rust traces of a fresh neuron each."""
    make = neuron_factory if callable(neuron_factory) else None
    assert make is not None
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        traces = [
            _verilog_trace(make(), "m", 0.0, steps, width, fraction, tmp),
            _c_trace(make(), "m", 0, steps, width, fraction, tmp),
        ]
        if HAS_RUST:
            traces.append(_rust_draw_trace(make(), steps, width, fraction, tmp))
    return traces


def _rust_draw_trace(
    neuron: EquationNeuron, steps: int, width: int, fraction: int, tmp: Path
) -> list[list[str]]:
    """Return the Rust kernel's trace; a drawing kernel's state also holds ``rng_step``."""
    kernel = generate_bittrue_kernel_from_neuron(
        neuron, "m", data_width=width, fraction=fraction, language="rust"
    )
    graded = ", spike_value_out: 0" if neuron.spike_value_expr else ""
    main = _rust_main(neuron, "m", 0, steps, width).replace(
        "spike_out: 0", f"spike_out: 0, rng_step: 0{graded}", 1
    )
    (tmp / "m.rs").write_text("#![allow(warnings)]\n" + kernel + "\n" + main)
    subprocess.run(
        ["rustc", "-C", "overflow-checks=on", "-o", str(tmp / "r"), str(tmp / "m.rs")],
        check=True,
        capture_output=True,
        timeout=300,
    )
    return _rows(subprocess.run([str(tmp / "r")], capture_output=True, text=True).stdout)


@pytest.mark.skipif(not HAS_COSIM, reason="Icarus Verilog / gcc not available")
def test_rtl_c_and_rust_draw_xi_as_the_reference() -> None:
    """Map v = xi at Q8.8: every target's v code is 2 · normal_code of XI_SITE at that step."""
    steps = 12
    traces = _traces(_map_neuron, steps, 16, 8)
    reference = [
        str(2 * ru.normal_code(*ru.draw(XI_SITE, step, 0, _SEED)[:2], 16, 8))
        for step in range(steps)
    ]
    for trace in traces:
        assert [row[1] for row in trace] == reference


@pytest.mark.skipif(not HAS_COSIM, reason="Icarus Verilog / gcc not available")
def test_the_exact_unit_lowers_xi_too() -> None:
    """With random_precision='exact' the RTL v code is 2 · normal_code_exact at Q3.29."""
    steps = 8
    rtl = compile_to_verilog(_map_neuron(), "m", 32, 28, random_precision="exact")
    assert "exact mode" in rtl
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        (tmp / "m.v").write_text(rtl)
        (tmp / "tb.v").write_text(_testbench(_map_neuron(), "m", 0.0, steps, 32, 28))
        subprocess.run(
            ["iverilog", "-g2012", "-o", str(tmp / "s"), str(tmp / "m.v"), str(tmp / "tb.v")],
            check=True,
            capture_output=True,
            timeout=120,
        )
        out = subprocess.run(["vvp", str(tmp / "s")], capture_output=True, text=True, timeout=120)
    rows = _rows(out.stdout)
    reference = [
        str(2 * xn.normal_code_exact(*ru.draw(XI_SITE, step, 0, _SEED)[:2], 32, 28))
        for step in range(steps)
    ]
    assert [row[1] for row in rows] == reference


def _euler_neuron() -> EquationNeuron:
    """Return an Ornstein-Uhlenbeck-like Euler neuron dv/dt = -v + 0.5·xi at dt = 0.25."""
    return EquationNeuron(
        equations={"v": "-v + 0.5 * xi"}, state={"v": 0.0}, dt=0.25, rng_seed=_SEED
    )


@pytest.mark.skipif(not HAS_COSIM, reason="Icarus Verilog / gcc not available")
def test_an_euler_xi_neuron_is_equal_on_every_target_and_near_the_model() -> None:
    """RTL = C = Rust word for word; the model stays within a few LSB of them."""
    steps = 40
    traces = _traces(_euler_neuron, steps, 16, 8)
    assert all(trace == traces[0] for trace in traces)
    model = _euler_neuron()
    for row in traces[0]:
        model.step(0.0)
        assert abs(int(row[1]) - model.state["v"] * 256) <= 4


def _reset_and_graded_neuron() -> EquationNeuron:
    """Return a neuron reading xi in its equation, its reset and its graded spike value."""
    return EquationNeuron(
        equations={"v": "-v + 4.0 + 0.5 * xi"},
        state={"v": 0.0},
        dt=0.25,
        threshold="v > 1.0",
        reset={"v": "0.25 * xi"},
        spike_value="1.0 + 0.125 * xi",
        rng_seed=_SEED,
    )


@pytest.mark.skipif(not HAS_COSIM, reason="Icarus Verilog / gcc not available")
def test_xi_in_reset_and_graded_value_compiles_equally_everywhere() -> None:
    """RTL = C = Rust word for word with xi in the equation, the reset and the spike value."""
    traces = _traces(_reset_and_graded_neuron, 40, 16, 8)
    assert all(trace == traces[0] for trace in traces)
    assert any(row[0] == "1" for row in traces[0]), "the neuron spikes and resets"

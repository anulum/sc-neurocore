# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Executed Wong-Wang five-runtime parity

"""Exercise full Euler/OU traces, public dispatch, and native failure atomicity."""

from __future__ import annotations

import ctypes
from pathlib import Path
import subprocess
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pytest

from sc_neurocore.accel import wong_wang as backends
from sc_neurocore.neurons.models.wong_wang import WongWangUnit

_REPOSITORY = Path(__file__).resolve().parents[1]
_COMPILED_BACKENDS = ("rust", "julia", "go", "mojo")
_CONFIG = {
    "s1": 0.24,
    "s2": 0.11,
    "noise1": 0.01,
    "noise2": -0.02,
    "tau_s": 0.12,
    "tau_ampa": 0.003,
    "gamma": 0.7,
    "j_n": 0.28,
    "j_cross": 0.06,
    "i_0": 0.31,
    "sigma": 0.015,
    "dt": 0.0002,
}


def _inputs(steps: int) -> tuple[npt.NDArray[np.float64], ...]:
    """Return varied finite stimuli and explicit normal samples."""
    index = np.arange(steps, dtype=np.float64)
    return (
        0.02 + 0.01 * np.sin(index * 0.07),
        -0.01 + 0.008 * np.cos(index * 0.11),
        np.sin(np.arange(2 * steps, dtype=np.float64) * 0.17),
    )


def _assert_result_close(
    actual: dict[str, npt.NDArray[np.float64] | float],
    expected: dict[str, npt.NDArray[np.float64] | float],
    tolerance: float,
) -> None:
    """Compare every trace and final-state field."""
    for key in ("s1", "s2", "noise1", "noise2", "r1", "r2"):
        np.testing.assert_allclose(actual[key], expected[key], rtol=0.0, atol=tolerance)
    for key in ("s1_final", "s2_final", "noise1_final", "noise2_final"):
        assert float(actual[key]) == pytest.approx(float(expected[key]), abs=tolerance)


@pytest.mark.parametrize("backend", ("python", *_COMPILED_BACKENDS))
def test_complete_configured_contract_matches_python(backend: str) -> None:
    """Execute all six traces and four final states on every maintained runtime."""
    stim1, stim2, xi = _inputs(128)
    expected = WongWangUnit(**_CONFIG).simulate(stim1, stim2, xi, backend="python")
    actual = WongWangUnit(**_CONFIG).simulate(stim1, stim2, xi, backend=backend)
    _assert_result_close(actual, expected, backends.PARITY_ATOL[backend])


@pytest.mark.parametrize("backend", ("python", *_COMPILED_BACKENDS))
def test_empty_batch_preserves_all_four_dynamic_states(backend: str) -> None:
    """Treat zero work as a complete backend contract, not a special skip."""
    unit = WongWangUnit(**_CONFIG)
    result = unit.simulate([], [], [], backend=backend)
    assert all(
        cast(npt.NDArray[np.float64], result[key]).shape == (0,)
        for key in ("s1", "s2", "noise1", "noise2", "r1", "r2")
    )
    assert (unit.s1, unit.s2, unit.noise1, unit.noise2) == (
        _CONFIG["s1"],
        _CONFIG["s2"],
        _CONFIG["noise1"],
        _CONFIG["noise2"],
    )


def test_auto_executes_a_maintained_lane() -> None:
    """Exercise the measured-order entry point without assuming one host winner."""
    stim1, stim2, xi = _inputs(4)
    unit = WongWangUnit()
    result = unit.simulate(stim1, stim2, xi)
    assert cast(npt.NDArray[np.float64], result["s1"]).shape == (4,)
    assert unit.s1 == result["s1_final"]


@pytest.mark.parametrize("backend", ("go", "mojo"))
def test_c_abi_invalid_configuration_writes_nothing(backend: str) -> None:
    """Reject an invalid scalar contract before touching caller buffers."""
    module = __import__(f"sc_neurocore.accel.{backend}.wong_wang", fromlist=["wong_wang"])
    assert bool(getattr(module, f"_HAS_{backend.upper()}_WONG_WANG"))
    library: Any = module._lib
    steps = 2
    stim1, stim2, xi = _inputs(steps)
    traces = [np.full(steps, -999.0, dtype=np.float64) for _ in range(6)]
    finals = [np.full(1, -999.0, dtype=np.float64) for _ in range(4)]
    if backend == "go":
        final_args: tuple[Any, ...] = tuple(
            value.ctypes.data_as(ctypes.POINTER(ctypes.c_double)) for value in finals
        )
    else:
        final_args = tuple(value.ctypes.data for value in finals)
    status = library.wong_wang_simulate_c(
        steps,
        0.1,
        0.1,
        0.0,
        0.0,
        0.1,
        0.002,
        0.641,
        0.2609,
        0.0497,
        0.3255,
        0.02,
        -0.0001,
        stim1.ctypes.data,
        stim2.ctypes.data,
        xi.ctypes.data,
        *(trace.ctypes.data for trace in traces),
        *final_args,
    )
    assert status == 2
    for trace in traces:
        np.testing.assert_array_equal(trace, np.full(steps, -999.0))
    for final in finals:
        np.testing.assert_array_equal(final, np.full(1, -999.0))


def test_standalone_rust_safety_matches_python_trace(tmp_path: Path) -> None:
    """Compile the dependency-free safety mirror and compare all physical states."""
    source = _REPOSITORY / "src/sc_neurocore/accel/rust/safety/wong_wang.rs"
    program = tmp_path / "wong_wang_trace.rs"
    binary = tmp_path / "wong_wang_trace"
    program.write_text(
        f'''include!(r#"{source}"#);

fn main() {{
    let mut state = WongWangUnit::new();
    for index in 0..16 {{
        let t = index as f64;
        let rates = state
            .step(0.02 + 0.01 * (t * 0.07).sin(), -0.01 + 0.008 * (t * 0.11).cos(),
                  (2.0 * t * 0.17).sin(), ((2.0 * t + 1.0) * 0.17).sin())
            .expect("valid Wong-Wang input");
        println!("{{:.17}} {{:.17}} {{:.17}} {{:.17}} {{:.17}} {{:.17}}",
                 state.s1, state.s2, state.noise1, state.noise2, rates.0, rates.1);
    }}
}}
''',
        encoding="utf-8",
    )
    subprocess.run(
        ["rustc", "--edition", "2021", "-O", str(program), "-o", str(binary)],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    completed = subprocess.run(
        [str(binary)], check=True, capture_output=True, text=True, timeout=30
    )
    actual = np.asarray(
        [[float(value) for value in line.split()] for line in completed.stdout.splitlines()]
    )
    reference = WongWangUnit()
    expected = []
    stim1, stim2, xi = _inputs(16)
    for index in range(16):
        rates = reference.step_with_gaussian_samples(
            stim1[index], stim2[index], xi[2 * index], xi[2 * index + 1]
        )
        expected.append((reference.s1, reference.s2, reference.noise1, reference.noise2, *rates))
    np.testing.assert_allclose(actual, np.asarray(expected), rtol=0.0, atol=1.0e-15)

# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Executed Jansen–Rit five-runtime parity

"""Exercise complete trajectories, native boundaries, and safety atomicity."""

from __future__ import annotations

import ctypes
from pathlib import Path
import subprocess
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pytest

from sc_neurocore.accel import jansen_rit as backends
from sc_neurocore.neurons.models.jansen_rit import JansenRitUnit

_REPOSITORY = Path(__file__).resolve().parents[1]
_COMPILED_BACKENDS = ("rust", "julia", "go", "mojo")
_STATE_KEYS = ("y0", "y3", "y1", "y4", "y2", "y5")
_TRACE_KEYS = (*_STATE_KEYS, "eeg")
_CONFIG = {
    "y0": 0.1,
    "y3": 0.2,
    "y1": 0.3,
    "y4": -0.4,
    "y2": -0.1,
    "y5": 0.5,
    "a_exc": 3.4,
    "b_exc": 21.0,
    "a_rate": 95.0,
    "b_rate": 55.0,
    "c": 128.0,
    "e0": 2.4,
    "v0": 5.8,
    "r": 0.6,
    "dt": 0.00012,
}


def _drive(steps: int) -> npt.NDArray[np.float64]:
    index = np.arange(steps, dtype=np.float64)
    return 220.0 + 80.0 * np.sin(index * 0.037) + 20.0 * np.cos(index * 0.011)


def _assert_result_close(
    actual: dict[str, npt.NDArray[np.float64] | float],
    expected: dict[str, npt.NDArray[np.float64] | float],
    tolerance: float,
) -> None:
    for key in _TRACE_KEYS:
        np.testing.assert_allclose(actual[key], expected[key], rtol=0.0, atol=tolerance)
    for key in _STATE_KEYS:
        assert float(actual[f"{key}_final"]) == pytest.approx(
            float(expected[f"{key}_final"]), abs=tolerance
        )


@pytest.mark.parametrize("backend", ("python", *_COMPILED_BACKENDS))
def test_complete_configured_contract_matches_python(backend: str) -> None:
    expected = JansenRitUnit(**_CONFIG).simulate(_drive(128), backend="python")
    actual = JansenRitUnit(**_CONFIG).simulate(_drive(128), backend=backend)
    _assert_result_close(actual, expected, backends.PARITY_ATOL[backend])


@pytest.mark.parametrize("backend", ("python", *_COMPILED_BACKENDS))
def test_empty_batch_preserves_all_six_dynamic_states(backend: str) -> None:
    unit = JansenRitUnit(**_CONFIG)
    result = unit.simulate([], backend=backend)
    assert all(cast(npt.NDArray[np.float64], result[key]).shape == (0,) for key in _TRACE_KEYS)
    assert tuple(getattr(unit, key) for key in _STATE_KEYS) == tuple(
        _CONFIG[key] for key in _STATE_KEYS
    )


def test_auto_executes_a_maintained_lane() -> None:
    unit = JansenRitUnit()
    result = unit.simulate(_drive(4))
    assert cast(npt.NDArray[np.float64], result["eeg"]).shape == (4,)
    assert unit.y5 == result["y5_final"]


@pytest.mark.parametrize("backend", ("go", "mojo"))
def test_c_abi_invalid_configuration_writes_nothing(backend: str) -> None:
    module = __import__(f"sc_neurocore.accel.{backend}.jansen_rit", fromlist=["jansen_rit"])
    assert bool(getattr(module, f"_HAS_{backend.upper()}_JANSEN_RIT"))
    library: Any = module._lib
    steps = 2
    drive = _drive(steps)
    traces = [np.full(steps, -999.0, dtype=np.float64) for _ in range(7)]
    finals = [np.full(1, -999.0, dtype=np.float64) for _ in range(6)]
    final_args: tuple[Any, ...]
    if backend == "go":
        final_args = tuple(
            value.ctypes.data_as(ctypes.POINTER(ctypes.c_double)) for value in finals
        )
    else:
        final_args = tuple(value.ctypes.data for value in finals)
    status = library.jansen_rit_simulate_c(
        steps,
        0.1,
        0.2,
        0.3,
        -0.4,
        -0.1,
        0.5,
        3.25,
        22.0,
        100.0,
        50.0,
        135.0,
        2.5,
        6.0,
        0.56,
        -0.0001,
        drive.ctypes.data,
        *(trace.ctypes.data for trace in traces),
        *final_args,
    )
    assert status == 2
    for trace in traces:
        np.testing.assert_array_equal(trace, np.full(steps, -999.0))
    for final in finals:
        np.testing.assert_array_equal(final, np.full(1, -999.0))


def test_standalone_rust_safety_matches_python_trace(tmp_path: Path) -> None:
    source = _REPOSITORY / "src/sc_neurocore/accel/rust/safety/jansen_rit.rs"
    program = tmp_path / "jansen_rit_trace.rs"
    binary = tmp_path / "jansen_rit_trace"
    program.write_text(
        f'''include!(r#"{source}"#);

fn main() {{
    let mut state = JansenRitUnit::new();
    for index in 0..16 {{
        let p_ext = 220.0 + 80.0 * ((index as f64) * 0.037).sin();
        let eeg = state.step(p_ext).expect("valid Jansen-Rit input");
        println!("{{:.17}} {{:.17}} {{:.17}} {{:.17}} {{:.17}} {{:.17}} {{:.17}}",
                 state.y0, state.y3, state.y1, state.y4, state.y2, state.y5, eeg);
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
    reference = JansenRitUnit()
    expected = []
    for index in range(16):
        eeg = reference.step(220.0 + 80.0 * np.sin(index * 0.037))
        expected.append(
            (
                reference.y0,
                reference.y3,
                reference.y1,
                reference.y4,
                reference.y2,
                reference.y5,
                eeg,
            )
        )
    np.testing.assert_allclose(actual, np.asarray(expected), rtol=0.0, atol=1.0e-15)

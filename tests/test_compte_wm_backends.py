# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — executed five-runtime Compte parity

"""Exercise complete Compte traces and native failure atomicity."""

from __future__ import annotations

import ctypes
import math
from pathlib import Path
import subprocess
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pytest

from sc_neurocore.accel import compte_wm as backends
from sc_neurocore.neurons.models.compte_wm import CompteWMNeuron

_ROOT = Path(__file__).resolve().parents[1]
_LANES = ("python", "rust", "julia", "go", "mojo")
_TRACE_KEYS = ("voltages", "s_ampa", "s_nmda", "x_nmda", "s_gaba", "refractory")


def _inputs(steps: int) -> tuple[npt.NDArray[np.float64], ...]:
    index = np.arange(steps)
    return (
        1.0 + 0.2 * np.sin(index * 0.03),
        (index % 17 == 0).astype(np.int64),
        (index % 11 == 0).astype(np.int64),
        (index % 23 == 0).astype(np.int64),
    )


@pytest.mark.parametrize("backend", _LANES)
def test_complete_configured_contract_matches_python(backend: str) -> None:
    """Compare all six state traces, final states, and sampled events."""
    expected = CompteWMNeuron().simulate(*_inputs(512), backend="python")
    actual = CompteWMNeuron().simulate(*_inputs(512), backend=backend)
    actual_events = cast(npt.NDArray[np.int64], actual["events"])
    expected_events = cast(npt.NDArray[np.int64], expected["events"])
    np.testing.assert_array_equal(actual_events, expected_events)
    assert int(np.sum(actual_events)) > 0
    for key in _TRACE_KEYS:
        np.testing.assert_allclose(
            cast(npt.NDArray[np.float64], actual[key]),
            cast(npt.NDArray[np.float64], expected[key]),
            rtol=0.0,
            atol=backends.PARITY_ATOL[backend],
        )
    for key in (
        "v_final",
        "s_ampa_final",
        "s_nmda_final",
        "x_nmda_final",
        "s_gaba_final",
        "ref_final",
    ):
        assert actual[key] == pytest.approx(expected[key], abs=backends.PARITY_ATOL[backend])


@pytest.mark.parametrize("backend", _LANES)
def test_empty_batch_preserves_complete_dynamic_state(backend: str) -> None:
    neuron = CompteWMNeuron(v=-63.0, s_ampa=0.2, s_nmda=0.1, x_nmda=0.3, s_gaba=0.4)
    neuron._ref_remaining = 0.7
    before = neuron.get_state()
    result = neuron.simulate([], [], [], [], backend=backend)
    assert cast(npt.NDArray[np.float64], result["voltages"]).shape == (0,)
    assert neuron.get_state() == before


@pytest.mark.parametrize("backend", ("go", "mojo"))
def test_c_abi_rejects_invalid_input_before_writing(backend: str) -> None:
    """Prove the native boundary validates every input before output writes."""
    module: Any = __import__(f"sc_neurocore.accel.{backend}.compte_wm", fromlist=["x"])
    library: Any = module._lib
    inputs = (
        np.array([math.nan], dtype=np.float64),
        np.zeros(1, dtype=np.int64),
        np.zeros(1, dtype=np.int64),
        np.zeros(1, dtype=np.int64),
    )
    traces = [np.full(1, -999.0) for _ in range(6)]
    events = np.full(1, -999, dtype=np.int64)
    finals = [np.full(1, -999.0) for _ in range(6)]
    address = ctypes.c_void_p if backend == "go" else ctypes.c_ssize_t
    config = (
        -70.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.025,
        0.0031,
        0.000381,
        0.001336,
        -70.0,
        0.0,
        -70.0,
        0.5,
        1.0,
        2.0,
        100.0,
        2.0,
        10.0,
        0.5,
        -50.0,
        -60.0,
        2.0,
        0.02,
    )
    status = library.compte_wm_simulate_c(
        1,
        *(ctypes.c_double(value) for value in config),
        *(address(value.ctypes.data) for value in inputs),
        *(address(value.ctypes.data) for value in traces),
        address(events.ctypes.data),
        *(address(value.ctypes.data) for value in finals),
    )
    assert status == 3
    assert [value[0] for value in (*traces, events, *finals)] == [-999.0] * 13


def test_dependency_free_rust_safety_matches_python(tmp_path: Path) -> None:
    """Compile the independent Rust file and compare a varied state receipt."""
    source = _ROOT / "src/sc_neurocore/accel/rust/safety/compte_wm.rs"
    program = tmp_path / "trace.rs"
    binary = tmp_path / "trace"
    program.write_text(
        f'''include!(r#"{source}"#);
fn main() {{
    let mut state = CompteWMNeuron::new();
    for i in 0..64 {{
        let event = state.step(
            1.0 + 0.2 * ((i as f64) * 0.03).sin(),
            i % 17 == 0, i % 11 == 0, i % 23 == 0,
        ).unwrap();
        let values = state.get_state();
        println!("{{:.17}} {{:.17}} {{:.17}} {{:.17}} {{:.17}} {{:.17}} {{}}",
            values[0], values[1], values[2], values[3], values[4], values[5], event);
    }}
}}
''',
        encoding="utf-8",
    )
    subprocess.run(
        ["rustc", "--edition", "2021", "-O", str(program), "-o", str(binary)], check=True
    )
    output = subprocess.run([str(binary)], check=True, capture_output=True, text=True).stdout
    actual = np.asarray([[float(value) for value in line.split()] for line in output.splitlines()])
    expected = CompteWMNeuron().simulate(*_inputs(64), backend="python")
    target: npt.NDArray[np.float64] = np.column_stack(
        (
            *(cast(npt.NDArray[np.float64], expected[key]) for key in _TRACE_KEYS),
            cast(npt.NDArray[np.int64], expected["events"]),
        )
    )
    np.testing.assert_allclose(actual, target, rtol=0.0, atol=2.0e-14)

# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — executed five-runtime Brunel-Wang parity

"""Exercise complete traces, configurable defaults, and native error atomicity."""

from __future__ import annotations

import ctypes
import math
from pathlib import Path
import subprocess
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pytest

from sc_neurocore.accel import brunel_wang as backends
from sc_neurocore.neurons.models.brunel_wang import BrunelWangNeuron

_REPOSITORY = Path(__file__).resolve().parents[1]
_LANES = ("python", "rust", "julia", "go", "mojo")


def _gates(steps: int) -> tuple[npt.NDArray[np.float64], ...]:
    index = np.arange(steps, dtype=np.float64)
    return (
        0.035 + 0.018 * (1.0 + np.sin(index * 0.071)),
        0.12 + 0.05 * (1.0 + np.cos(index * 0.053)),
        0.08 + 0.04 * (1.0 + np.sin(index * 0.037 + 0.2)),
        0.03 + 0.02 * (1.0 + np.cos(index * 0.089)),
    )


@pytest.mark.parametrize("backend", _LANES)
def test_complete_configured_contract_matches_python(backend: str) -> None:
    """Compare voltage, refractory, event, and final-state observables."""
    gates = _gates(256)
    expected = BrunelWangNeuron().simulate(*gates, backend="python")
    actual = BrunelWangNeuron().simulate(*gates, backend=backend)
    np.testing.assert_array_equal(actual["events"], expected["events"])
    np.testing.assert_allclose(
        actual["voltages"], expected["voltages"], rtol=0.0, atol=backends.PARITY_ATOL[backend]
    )
    np.testing.assert_allclose(
        actual["refractory"], expected["refractory"], rtol=0.0, atol=backends.PARITY_ATOL[backend]
    )
    assert actual["v_final"] == pytest.approx(
        expected["v_final"], abs=backends.PARITY_ATOL[backend]
    )
    assert actual["ref_final"] == pytest.approx(
        expected["ref_final"], abs=backends.PARITY_ATOL[backend]
    )


@pytest.mark.parametrize("backend", _LANES)
def test_empty_batch_preserves_dynamic_state(backend: str) -> None:
    """Make zero work a complete state-preserving backend contract."""
    neuron = BrunelWangNeuron(v=-63.0)
    neuron._ref_remaining = 0.7
    result = neuron.simulate([], [], [], [], backend=backend)
    assert cast(npt.NDArray[np.float64], result["voltages"]).shape == (0,)
    assert (neuron.v, neuron._ref_remaining) == (-63.0, 0.7)


@pytest.mark.parametrize("backend", ("go", "mojo"))
def test_c_abi_rejects_invalid_gate_before_writing(backend: str) -> None:
    """Reject NaN input without exposing partially written output buffers."""
    module: Any = __import__(f"sc_neurocore.accel.{backend}.brunel_wang", fromlist=["x"])
    library: Any = module._lib
    gates = [np.array([math.nan], dtype=np.float64)] + [np.zeros(1) for _ in range(3)]
    voltages = np.full(1, -999.0)
    refractory = np.full(1, -999.0)
    events = np.full(1, -999, dtype=np.int64)
    finals = [np.full(1, -999.0), np.full(1, -999.0)]
    address = ctypes.c_void_p if backend == "go" else ctypes.c_ssize_t
    status = library.brunel_wang_simulate_c(
        1,
        *(
            ctypes.c_double(value)
            for value in (
                -70.0,
                0.0,
                -70.0,
                -55.0,
                -50.0,
                20.0,
                2.0,
                2.08,
                0.104,
                0.327,
                1.25,
                0.0,
                0.0,
                -70.0,
                0.5,
                1.0,
                0.1,
            )
        ),
        *(address(array.ctypes.data) for array in gates),
        address(voltages.ctypes.data),
        address(refractory.ctypes.data),
        address(events.ctypes.data),
        *(address(array.ctypes.data) for array in finals),
    )
    assert status == 3
    assert (voltages[0], refractory[0], events[0], finals[0][0], finals[1][0]) == (
        -999.0,
        -999.0,
        -999,
        -999.0,
        -999.0,
    )


def test_standalone_rust_safety_matches_python(tmp_path: Path) -> None:
    """Compile the dependency-free Rust mirror and compare a varied trace."""
    source = _REPOSITORY / "src/sc_neurocore/accel/rust/safety/brunel_wang.rs"
    program = tmp_path / "trace.rs"
    binary = tmp_path / "trace"
    program.write_text(
        f'''include!(r#"{source}"#);
fn main() {{
    let mut state = BrunelWangNeuron::new();
    for i in 0..32 {{
        let t = i as f64;
        let e = state.step(0.035+0.018*(1.0+(t*0.071).sin()),
            0.12+0.05*(1.0+(t*0.053).cos()),
            0.08+0.04*(1.0+(t*0.037+0.2).sin()),
            0.03+0.02*(1.0+(t*0.089).cos())).unwrap();
        println!("{{:.17}} {{:.17}} {{}}", state.v, state.ref_remaining, e);
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
    expected = BrunelWangNeuron().simulate(*_gates(32), backend="python")
    target = np.column_stack((expected["voltages"], expected["refractory"], expected["events"]))
    np.testing.assert_allclose(actual, target, rtol=0.0, atol=2.0e-14)

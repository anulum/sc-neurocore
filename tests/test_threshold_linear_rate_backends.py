# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Executed threshold-linear rate backend parity

"""Exercise complete configured contracts through every maintained runtime."""

from __future__ import annotations

import ctypes
from pathlib import Path
import subprocess
from typing import Any

import numpy as np
import pytest

from sc_neurocore.accel import threshold_linear_rate as backends
from sc_neurocore.neurons.models.threshold_linear_rate import ThresholdLinearRateNeuron

_REPOSITORY = Path(__file__).resolve().parents[1]
_CONFIG = {"r": 0.25, "theta": 1.5, "gain": 2.0}
_COMPILED_BACKENDS = ("rust", "julia", "go", "mojo")


@pytest.mark.parametrize("backend", ("python", *_COMPILED_BACKENDS))
def test_complete_configured_contract_matches_python(backend: str) -> None:
    reference_neuron = ThresholdLinearRateNeuron(**_CONFIG)
    reference = reference_neuron.simulate(64, 3.0, backend="python")
    actual_neuron = ThresholdLinearRateNeuron(**_CONFIG)
    actual = actual_neuron.simulate(64, 3.0, backend=backend)
    np.testing.assert_array_equal(actual, reference)
    assert actual_neuron.r == reference_neuron.r == 3.0


@pytest.mark.parametrize("backend", ("python", *_COMPILED_BACKENDS))
@pytest.mark.parametrize(("current", "expected"), [(1.0, 0.0), (1.5, 0.0), (3.0, 3.0)])
def test_threshold_branches_are_exact(backend: str, current: float, expected: float) -> None:
    neuron = ThresholdLinearRateNeuron(**_CONFIG)
    trace = neuron.simulate(4, current, backend=backend)
    np.testing.assert_array_equal(trace, np.full(4, expected))
    assert neuron.r == expected


@pytest.mark.parametrize("backend", ("python", *_COMPILED_BACKENDS))
def test_empty_batch_preserves_cached_output(backend: str) -> None:
    neuron = ThresholdLinearRateNeuron(**_CONFIG)
    np.testing.assert_array_equal(neuron.simulate(0, 3.0, backend=backend), np.empty(0))
    assert neuron.r == 0.25


def test_auto_executes_a_maintained_lane() -> None:
    neuron = ThresholdLinearRateNeuron(**_CONFIG)
    np.testing.assert_array_equal(neuron.simulate(4, 3.0), np.full(4, 3.0))
    assert neuron.r == 3.0


@pytest.mark.parametrize("backend", _COMPILED_BACKENDS)
def test_requested_backend_reports_unavailable(
    backend: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if backend == "rust":
        monkeypatch.setattr(backends, "_HAS_RUST", False)
        monkeypatch.setattr(backends, "_engine_simulate", None)
    else:
        monkeypatch.setattr(backends, f"ensure_{backend}_loaded", lambda: False)
    neuron = ThresholdLinearRateNeuron(**_CONFIG)
    with pytest.raises(RuntimeError, match=backend.title()):
        neuron.simulate(1, 3.0, backend=backend)
    assert neuron.r == 0.25


class _RejectingCAbi:
    def threshold_linear_rate_simulate_c(self, *_args: object) -> int:
        return -1


def test_public_c_runners_name_rejected_contracts(monkeypatch: pytest.MonkeyPatch) -> None:
    rejecting = _RejectingCAbi()
    monkeypatch.setattr(backends, "_go_lib", rejecting)
    monkeypatch.setattr(backends, "_mojo_lib", rejecting)
    args = (0.25, 1.5, 2.0, 1, 3.0)
    with pytest.raises(FloatingPointError, match="Go ThresholdLinearRate kernel rejected"):
        backends.simulate_go(*args)
    with pytest.raises(FloatingPointError, match="Mojo ThresholdLinearRate kernel rejected"):
        backends.simulate_mojo(*args)


@pytest.mark.parametrize("backend", ("go", "mojo"))
def test_c_abi_invalid_contract_writes_nothing(backend: str) -> None:
    assert {"go": backends.ensure_go_loaded, "mojo": backends.ensure_mojo_loaded}[backend]()
    library: Any = backends._go_lib if backend == "go" else backends._mojo_lib
    output = np.full(2, -999.0, dtype=np.float64)
    destination: Any = (
        output.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        if backend == "go"
        else int(output.ctypes.data)
    )
    status = library.threshold_linear_rate_simulate_c(
        0.25,
        1.5,
        -2.0,
        1,
        3.0,
        destination,
    )
    assert status == -1
    np.testing.assert_array_equal(output, np.full(2, -999.0))


def test_standalone_rust_safety_matches_python_trace(tmp_path: Path) -> None:
    source = _REPOSITORY / "src/sc_neurocore/accel/rust/safety/threshold_linear_rate.rs"
    program = tmp_path / "threshold_linear_rate_trace.rs"
    binary = tmp_path / "threshold_linear_rate_trace"
    program.write_text(
        f'''include!(r#"{source}"#);

fn main() {{
    let mut state = ThresholdLinearRateNeuron::with_parameters(0.25, 1.5, 2.0)
        .expect("valid threshold-linear contract");
    for _ in 0..6 {{
        println!("{{:.17}}", state.step(3.0).expect("valid threshold-linear input"));
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
    actual = np.asarray([float(value) for value in completed.stdout.splitlines()])
    reference = ThresholdLinearRateNeuron(**_CONFIG).simulate(6, 3.0, backend="python")
    np.testing.assert_array_equal(actual, reference)

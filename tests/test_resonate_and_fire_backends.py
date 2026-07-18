# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Executed resonate-and-fire five-runtime parity

"""Exercise complete trajectories, native boundaries, and safety atomicity."""

from __future__ import annotations

from pathlib import Path
import subprocess
from typing import cast

import numpy as np
import numpy.typing as npt
import pytest

from sc_neurocore.accel import resonate_and_fire as backends
from sc_neurocore.neurons.models.resonate_and_fire import ResonateAndFireNeuron

_REPOSITORY = Path(__file__).resolve().parents[1]
_COMPILED_BACKENDS = ("rust", "julia", "go", "mojo")
_TRACE_KEYS = ("x", "y", "spikes")
_CONFIG = {
    "x": 0.13,
    "y": -0.27,
    "b": -0.8,
    "omega": 7.5,
    "threshold": 0.9,
    "dt": 0.006,
}
_CANDIDATE_FAILURE_CONFIG = {
    "x": 0.25,
    "y": -0.5,
    "b": 1_000.0,
    "omega": 1.0,
    "threshold": 1.0e300,
    "dt": 1.0,
}


def _drive(steps: int) -> npt.NDArray[np.float64]:
    index: npt.NDArray[np.float64] = np.arange(steps, dtype=np.float64)
    return 4.5 + 1.4 * np.sin(index * 0.037) + 0.3 * np.cos(index * 0.011)


def _assert_result_close(
    actual: dict[str, npt.NDArray[np.float64] | float | int],
    expected: dict[str, npt.NDArray[np.float64] | float | int],
    tolerance: float,
) -> None:
    for key in _TRACE_KEYS:
        np.testing.assert_allclose(actual[key], expected[key], rtol=0.0, atol=tolerance)
    assert float(actual["x_final"]) == pytest.approx(float(expected["x_final"]), abs=tolerance)
    assert float(actual["y_final"]) == pytest.approx(float(expected["y_final"]), abs=tolerance)
    assert int(actual["spike_count"]) == int(expected["spike_count"])


@pytest.mark.parametrize("backend", ("python", *_COMPILED_BACKENDS))
def test_complete_configured_contract_matches_python(backend: str) -> None:
    expected = ResonateAndFireNeuron(**_CONFIG).simulate(_drive(128), backend="python")
    actual = ResonateAndFireNeuron(**_CONFIG).simulate(_drive(128), backend=backend)
    _assert_result_close(actual, expected, backends.PARITY_ATOL[backend])


@pytest.mark.parametrize("backend", ("python", *_COMPILED_BACKENDS))
def test_empty_batch_preserves_both_dynamic_states(backend: str) -> None:
    unit = ResonateAndFireNeuron(**_CONFIG)
    result = unit.simulate([], backend=backend)
    assert all(cast(npt.NDArray[np.float64], result[key]).shape == (0,) for key in _TRACE_KEYS)
    assert result["x_final"] == _CONFIG["x"]
    assert result["y_final"] == _CONFIG["y"]
    assert result["spike_count"] == 0
    assert (unit.x, unit.y) == (_CONFIG["x"], _CONFIG["y"])


def test_auto_executes_a_maintained_lane() -> None:
    unit = ResonateAndFireNeuron()
    result = unit.simulate(_drive(4))
    assert cast(npt.NDArray[np.float64], result["x"]).shape == (4,)
    assert unit.y == result["y_final"]


@pytest.mark.parametrize("backend", ("python", *_COMPILED_BACKENDS))
def test_candidate_failure_raises_floating_point_error_atomically(backend: str) -> None:
    unit = ResonateAndFireNeuron(**_CANDIDATE_FAILURE_CONFIG)
    before = (unit.x, unit.y)
    with pytest.raises(FloatingPointError):
        unit.simulate([0.0], backend=backend)
    assert (unit.x, unit.y) == before


def test_standalone_rust_safety_matches_python_trace(tmp_path: Path) -> None:
    source = _REPOSITORY / "src/sc_neurocore/accel/rust/safety/resonate_and_fire.rs"
    program = tmp_path / "resonate_and_fire_trace.rs"
    binary = tmp_path / "resonate_and_fire_trace"
    program.write_text(
        f'''include!(r#"{source}"#);

fn main() {{
    let mut state = ResonateAndFireNeuron::new();
    for index in 0..16 {{
        let drive = 4.5 + 1.4 * ((index as f64) * 0.037).sin();
        let spike = state.step(drive).expect("valid resonate-and-fire input");
        println!("{{:.17}} {{:.17}} {{}}", state.x, state.y, spike);
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
        [str(binary)],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    actual = np.asarray(
        [[float(value) for value in line.split()] for line in completed.stdout.splitlines()]
    )
    reference = ResonateAndFireNeuron()
    expected = []
    for index in range(16):
        spike = reference.step(4.5 + 1.4 * np.sin(index * 0.037))
        expected.append((reference.x, reference.y, spike))
    np.testing.assert_allclose(actual, np.asarray(expected), rtol=0.0, atol=1.0e-15)

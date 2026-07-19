# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Executed adaptive-threshold five-runtime parity

"""Exercise complete trajectories, native boundaries, and safety atomicity."""

from __future__ import annotations

from pathlib import Path
import subprocess
from typing import cast

import numpy as np
import numpy.typing as npt
import pytest

from sc_neurocore.accel import adaptive_threshold_if as backends
from sc_neurocore.neurons.models.adaptive_threshold_if import AdaptiveThresholdIFNeuron

_REPOSITORY = Path(__file__).resolve().parents[1]
_COMPILED_BACKENDS = ("rust", "julia", "go", "mojo")
_TRACE_KEYS = ("v", "theta", "spikes")
_CONFIG = {
    "v": -63.5,
    "theta": -52.5,
    "v_rest": -68.0,
    "v_reset": -67.0,
    "theta_rest": -49.0,
    "delta_theta": 4.5,
    "tau_m": 8.0,
    "tau_theta": 42.0,
    "dt": 0.05,
}
_CANDIDATE_FAILURE_CONFIG = {
    "v": -1.0e308,
    "theta": -45.0,
    "v_rest": -65.0,
    "v_reset": -65.0,
    "theta_rest": -50.0,
    "delta_theta": 5.0,
    "tau_m": 10.0,
    "tau_theta": 50.0,
    "dt": 0.1,
}


def _drive(steps: int) -> npt.NDArray[np.float64]:
    index: npt.NDArray[np.float64] = np.arange(steps, dtype=np.float64)
    return 22.0 + 6.0 * np.sin(index * 0.037) + 1.5 * np.cos(index * 0.011)


def _assert_result_close(
    actual: dict[str, npt.NDArray[np.float64] | float | int],
    expected: dict[str, npt.NDArray[np.float64] | float | int],
    tolerance: float,
) -> None:
    for key in _TRACE_KEYS:
        np.testing.assert_allclose(actual[key], expected[key], rtol=0.0, atol=tolerance)
    assert float(actual["v_final"]) == pytest.approx(float(expected["v_final"]), abs=tolerance)
    assert float(actual["theta_final"]) == pytest.approx(
        float(expected["theta_final"]), abs=tolerance
    )
    assert int(actual["spike_count"]) == int(expected["spike_count"])


@pytest.mark.parametrize("backend", ("python", *_COMPILED_BACKENDS))
def test_complete_configured_contract_matches_python(backend: str) -> None:
    expected = AdaptiveThresholdIFNeuron(**_CONFIG).simulate(_drive(256), backend="python")
    actual = AdaptiveThresholdIFNeuron(**_CONFIG).simulate(_drive(256), backend=backend)
    _assert_result_close(actual, expected, backends.PARITY_ATOL[backend])


@pytest.mark.parametrize("backend", ("python", *_COMPILED_BACKENDS))
def test_empty_batch_preserves_both_dynamic_states(backend: str) -> None:
    unit = AdaptiveThresholdIFNeuron(**_CONFIG)
    result = unit.simulate([], backend=backend)
    assert all(cast(npt.NDArray[np.float64], result[key]).shape == (0,) for key in _TRACE_KEYS)
    assert result["v_final"] == _CONFIG["v"]
    assert result["theta_final"] == _CONFIG["theta"]
    assert result["spike_count"] == 0
    assert (unit.v, unit.theta) == (_CONFIG["v"], _CONFIG["theta"])


def test_auto_executes_a_maintained_lane() -> None:
    unit = AdaptiveThresholdIFNeuron()
    result = unit.simulate(_drive(4))
    assert cast(npt.NDArray[np.float64], result["v"]).shape == (4,)
    assert unit.theta == result["theta_final"]


@pytest.mark.parametrize("backend", ("python", *_COMPILED_BACKENDS))
def test_candidate_failure_raises_floating_point_error_atomically(backend: str) -> None:
    unit = AdaptiveThresholdIFNeuron(**_CANDIDATE_FAILURE_CONFIG)
    before = (unit.v, unit.theta)
    with pytest.raises(FloatingPointError):
        unit.simulate([1.0e308], backend=backend)
    assert (unit.v, unit.theta) == before


def test_standalone_rust_safety_matches_python_trace(tmp_path: Path) -> None:
    source = _REPOSITORY / "src/sc_neurocore/accel/rust/safety/adaptive_threshold_if.rs"
    program = tmp_path / "adaptive_threshold_if_trace.rs"
    binary = tmp_path / "adaptive_threshold_if_trace"
    program.write_text(
        f'''include!(r#"{source}"#);

fn main() {{
    let mut state = AdaptiveThresholdIFNeuron::new();
    for index in 0..16 {{
        let drive = 22.0 + 6.0 * ((index as f64) * 0.037).sin();
        let spike = state.step(drive).expect("valid adaptive-threshold input");
        println!("{{:.17}} {{:.17}} {{}}", state.v, state.theta, spike);
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
    reference = AdaptiveThresholdIFNeuron()
    expected = []
    for index in range(16):
        spike = reference.step(22.0 + 6.0 * np.sin(index * 0.037))
        expected.append((reference.v, reference.theta, spike))
    np.testing.assert_allclose(actual, np.asarray(expected), rtol=0.0, atol=1.0e-15)

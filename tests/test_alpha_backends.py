# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Executed alpha-synapse five-runtime parity

"""Exercise complete trajectories, native boundaries, and safety atomicity."""

from __future__ import annotations

from pathlib import Path
import subprocess
from typing import cast

import numpy as np
import numpy.typing as npt
import pytest

from sc_neurocore.accel import alpha as backends
from sc_neurocore.neurons.models.alpha import AlphaNeuron

_REPOSITORY = Path(__file__).resolve().parents[1]
_COMPILED_BACKENDS = ("rust", "julia", "go", "mojo")
_TRACE_KEYS = ("v", "a_exc", "i_exc", "a_inh", "i_inh", "spikes")
_CONFIG = {
    "v": 0.15,
    "a_exc": 0.08,
    "i_exc": 0.05,
    "a_inh": 0.04,
    "i_inh": 0.03,
    "v_rest": -0.5,
    "v_threshold": 1.2,
    "tau_v": 16.0,
    "tau_exc": 4.0,
    "tau_inh": 9.0,
    "dt": 0.5,
}
_CANDIDATE_FAILURE_CONFIG = {
    "v": -1.0e308,
    "a_exc": 0.0,
    "i_exc": 0.0,
    "a_inh": 0.0,
    "i_inh": 0.0,
    "v_rest": 0.0,
    "v_threshold": 1.0,
    "tau_v": 20.0,
    "tau_exc": 5.0,
    "tau_inh": 10.0,
    "dt": 1.0,
}


def _drive(steps: int) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    index: npt.NDArray[np.float64] = np.arange(steps, dtype=np.float64)
    return (
        2.0 + 0.8 * np.sin(index * 0.037),
        0.7 + 0.3 * np.cos(index * 0.021),
    )


def _assert_result_close(
    actual: dict[str, npt.NDArray[np.float64] | float | int],
    expected: dict[str, npt.NDArray[np.float64] | float | int],
    tolerance: float,
) -> None:
    for key in _TRACE_KEYS:
        np.testing.assert_allclose(actual[key], expected[key], rtol=0.0, atol=tolerance)
    for key in ("v_final", "a_exc_final", "i_exc_final", "a_inh_final", "i_inh_final"):
        assert float(actual[key]) == pytest.approx(float(expected[key]), abs=tolerance)
    assert int(actual["spike_count"]) == int(expected["spike_count"])


@pytest.mark.parametrize("backend", ("python", *_COMPILED_BACKENDS))
def test_complete_configured_contract_matches_python(backend: str) -> None:
    exc, inh = _drive(256)
    expected = AlphaNeuron(**_CONFIG).simulate(exc, inh, backend="python")
    actual = AlphaNeuron(**_CONFIG).simulate(exc, inh, backend=backend)
    _assert_result_close(actual, expected, backends.PARITY_ATOL[backend])


@pytest.mark.parametrize("backend", ("python", *_COMPILED_BACKENDS))
def test_empty_batch_preserves_all_five_states(backend: str) -> None:
    unit = AlphaNeuron(**_CONFIG)
    result = unit.simulate([], backend=backend)
    assert all(cast(npt.NDArray[np.float64], result[key]).shape == (0,) for key in _TRACE_KEYS)
    assert result["v_final"] == _CONFIG["v"]
    assert result["a_exc_final"] == _CONFIG["a_exc"]
    assert result["spike_count"] == 0
    assert (unit.v, unit.a_exc) == (_CONFIG["v"], _CONFIG["a_exc"])


def test_auto_executes_a_maintained_lane() -> None:
    unit = AlphaNeuron()
    exc, inh = _drive(4)
    result = unit.simulate(exc, inh)
    assert cast(npt.NDArray[np.float64], result["v"]).shape == (4,)
    assert unit.v == result["v_final"]


@pytest.mark.parametrize("backend", ("python", *_COMPILED_BACKENDS))
def test_candidate_failure_raises_floating_point_error_atomically(backend: str) -> None:
    unit = AlphaNeuron(**_CANDIDATE_FAILURE_CONFIG)
    before = (unit.v, unit.a_exc, unit.i_exc, unit.a_inh, unit.i_inh)
    with pytest.raises(FloatingPointError):
        unit.simulate([1.0e308], backend=backend)
    assert (unit.v, unit.a_exc, unit.i_exc, unit.a_inh, unit.i_inh) == before


def test_standalone_rust_safety_matches_python_trace(tmp_path: Path) -> None:
    source = _REPOSITORY / "src/sc_neurocore/accel/rust/safety/alpha.rs"
    program = tmp_path / "alpha_trace.rs"
    binary = tmp_path / "alpha_trace"
    program.write_text(
        f'''include!(r#"{source}"#);

fn main() {{
    let mut state = AlphaNeuron::new();
    for index in 0..16 {{
        let exc = 2.0 + 0.8 * ((index as f64) * 0.037).sin();
        let inh = 0.7 + 0.3 * ((index as f64) * 0.021).cos();
        let spike = state.step(exc, inh).expect("valid alpha input");
        println!("{{:.17}} {{:.17}} {{:.17}} {{}}", state.v, state.a_exc, state.i_exc, spike);
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
    reference = AlphaNeuron()
    expected = []
    for index in range(16):
        spike = reference.step(2.0 + 0.8 * np.sin(index * 0.037), 0.7 + 0.3 * np.cos(index * 0.021))
        expected.append((reference.v, reference.a_exc, reference.i_exc, spike))
    np.testing.assert_allclose(actual, np.asarray(expected), rtol=0.0, atol=1.0e-15)

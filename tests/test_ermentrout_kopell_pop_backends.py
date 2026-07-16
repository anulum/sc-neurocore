# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Executed MPR five-runtime parity

"""Exercise complete trajectories, native boundaries, and safety atomicity."""

from __future__ import annotations

from pathlib import Path
import subprocess
from typing import cast

import numpy as np
import numpy.typing as npt
import pytest

from sc_neurocore.accel import ermentrout_kopell_pop as backends
from sc_neurocore.neurons.models.ermentrout_kopell_pop import (
    ErmentroutKopellPopulation,
)

_REPOSITORY = Path(__file__).resolve().parents[1]
_COMPILED_BACKENDS = ("rust", "julia", "go", "mojo")
_STATE_KEYS = ("r", "v")
_CONFIG = {
    "r": 0.13,
    "v": -1.7,
    "tau": 1.3,
    "delta": 0.8,
    "eta_bar": -4.2,
    "j": 12.5,
    "dt": 0.004,
}
_CANDIDATE_FAILURE_CONFIG = {
    "r": 1.0,
    "v": -100.0,
    "tau": 1.0,
    "delta": 0.0,
    "eta_bar": 0.0,
    "j": 0.0,
    "dt": 0.1,
}


def _drive(steps: int) -> npt.NDArray[np.float64]:
    index: npt.NDArray[np.float64] = np.arange(steps, dtype=np.float64)
    return 1.5 + 0.5 * np.sin(index * 0.037) + 0.25 * np.cos(index * 0.011)


def _assert_result_close(
    actual: dict[str, npt.NDArray[np.float64] | float],
    expected: dict[str, npt.NDArray[np.float64] | float],
    tolerance: float,
) -> None:
    for key in _STATE_KEYS:
        np.testing.assert_allclose(actual[key], expected[key], rtol=0.0, atol=tolerance)
        assert float(actual[f"{key}_final"]) == pytest.approx(
            float(expected[f"{key}_final"]), abs=tolerance
        )


@pytest.mark.parametrize("backend", ("python", *_COMPILED_BACKENDS))
def test_complete_configured_contract_matches_python(backend: str) -> None:
    expected = ErmentroutKopellPopulation(**_CONFIG).simulate(_drive(128), backend="python")
    actual = ErmentroutKopellPopulation(**_CONFIG).simulate(_drive(128), backend=backend)
    _assert_result_close(actual, expected, backends.PARITY_ATOL[backend])


@pytest.mark.parametrize("backend", ("python", *_COMPILED_BACKENDS))
def test_empty_batch_preserves_both_dynamic_states(backend: str) -> None:
    unit = ErmentroutKopellPopulation(**_CONFIG)
    result = unit.simulate([], backend=backend)
    assert all(cast(npt.NDArray[np.float64], result[key]).shape == (0,) for key in _STATE_KEYS)
    assert result["r_final"] == _CONFIG["r"]
    assert result["v_final"] == _CONFIG["v"]
    assert (unit.r, unit.v) == (_CONFIG["r"], _CONFIG["v"])


def test_auto_executes_a_maintained_lane() -> None:
    unit = ErmentroutKopellPopulation()
    result = unit.simulate(_drive(4))
    assert cast(npt.NDArray[np.float64], result["r"]).shape == (4,)
    assert unit.v == result["v_final"]


@pytest.mark.parametrize("backend", ("python", *_COMPILED_BACKENDS))
def test_candidate_failure_raises_floating_point_error_atomically(backend: str) -> None:
    unit = ErmentroutKopellPopulation(**_CANDIDATE_FAILURE_CONFIG)
    before = (unit.r, unit.v)
    with pytest.raises(FloatingPointError):
        unit.simulate([0.0], backend=backend)
    assert (unit.r, unit.v) == before


def test_standalone_rust_safety_matches_python_trace(tmp_path: Path) -> None:
    source = _REPOSITORY / "src/sc_neurocore/accel/rust/safety/ermentrout_kopell_pop.rs"
    program = tmp_path / "ermentrout_kopell_pop_trace.rs"
    binary = tmp_path / "ermentrout_kopell_pop_trace"
    program.write_text(
        f'''include!(r#"{source}"#);

fn main() {{
    let mut state = ErmentroutKopellPopulation::new();
    for index in 0..16 {{
        let drive = 1.5 + 0.5 * ((index as f64) * 0.037).sin();
        state.step(drive).expect("valid MPR input");
        println!("{{:.17}} {{:.17}}", state.r, state.v);
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
    reference = ErmentroutKopellPopulation()
    expected = []
    for index in range(16):
        reference.step(1.5 + 0.5 * np.sin(index * 0.037))
        expected.append((reference.r, reference.v))
    np.testing.assert_allclose(actual, np.asarray(expected), rtol=0.0, atol=1.0e-15)

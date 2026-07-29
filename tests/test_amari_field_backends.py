# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — executed five-runtime Amari field parity

"""Exercise complete vector state/rate receipts and failure contracts."""

from __future__ import annotations

from pathlib import Path
import subprocess

import numpy as np
import numpy.typing as npt
import pytest

from sc_neurocore.accel import amari_field as backends

ROOT = Path(__file__).resolve().parents[1]
RUNTIMES = ("python", "rust", "julia", "go", "mojo")
STATE = np.linspace(-0.2, 0.2, 8, dtype=np.float64)


def _drive(steps: int) -> npt.NDArray[np.float64]:
    index = np.arange(steps, dtype=np.float64)
    return np.column_stack([0.12 * np.sin(index * 0.17 + site * 0.31) - 0.01 for site in range(8)])


@pytest.mark.parametrize("backend", RUNTIMES)
def test_complete_vector_contract_matches_python(backend: str) -> None:
    expected = backends.simulate_amari_field(STATE, currents=_drive(128), backend="python")
    actual = backends.simulate_amari_field(STATE, currents=_drive(128), backend=backend)
    np.testing.assert_allclose(
        actual["states"], expected["states"], rtol=0.0, atol=backends.PARITY_ATOL[backend]
    )
    np.testing.assert_array_equal(actual["mean_rates"], expected["mean_rates"])
    np.testing.assert_allclose(
        actual["final_state"],
        expected["final_state"],
        rtol=0.0,
        atol=backends.PARITY_ATOL[backend],
    )


@pytest.mark.parametrize("backend", RUNTIMES)
def test_empty_batch_preserves_complete_initial_state(backend: str) -> None:
    result = backends.simulate_amari_field(
        STATE, currents=np.empty((0, 8), dtype=np.float64), backend=backend
    )
    assert result["states"].shape == (0, 8)
    assert result["mean_rates"].shape == (0,)
    np.testing.assert_array_equal(result["final_state"], STATE)


def test_scalar_drive_is_broadcast_across_the_field() -> None:
    scalar = backends.simulate_amari_field(STATE, currents=[0.1, -0.1], backend="python")
    vector = backends.simulate_amari_field(
        STATE, currents=np.asarray([[0.1] * 8, [-0.1] * 8]), backend="python"
    )
    for key in ("states", "mean_rates", "final_state"):
        np.testing.assert_array_equal(scalar[key], vector[key])


def test_nonfinite_input_is_rejected_before_native_dispatch() -> None:
    drive = _drive(2)
    drive[1, 4] = np.nan
    with pytest.raises(ValueError, match="finite"):
        backends.simulate_amari_field(STATE, currents=drive, backend="mojo")


def test_ci_builds_self_contained_go_and_mojo_libraries() -> None:
    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    build_loop = next(
        line for line in workflow.splitlines() if "for model in" in line and "wilson_cowan" in line
    )
    assert " amari_field " in f" {build_loop} "


def test_standalone_rust_safety_matches_python(tmp_path: Path) -> None:
    source = ROOT / "src/sc_neurocore/accel/rust/safety/amari_field.rs"
    program = tmp_path / "amari_trace.rs"
    binary = tmp_path / "amari_trace"
    program.write_text(
        f'''include!(r#"{source}"#);
fn main() {{
    let mut field = AmariNeuralField::with_config(
        (0..8).map(|site| -0.2 + (site as f64) * (0.4 / 7.0)).collect(),
        10.0, 1.5, 2.0, 0.75, 1.0, 0.5, 0.5,
    ).unwrap();
    for step in 0..16 {{
        let input: Vec<f64> = (0..8).map(|site|
            0.12 * ((step as f64) * 0.17 + (site as f64) * 0.31).sin() - 0.01
        ).collect();
        let rate = field.step(&input).unwrap();
        for value in &field.u {{ print!("{{:.17}} ", value); }}
        println!("{{:.17}}", rate);
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
    observed = subprocess.run(
        [str(binary)],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    actual = np.asarray(
        [[float(value) for value in line.split()] for line in observed.stdout.splitlines()]
    )
    expected = backends.simulate_amari_field(STATE, currents=_drive(16), backend="python")
    reference = np.column_stack((expected["states"], expected["mean_rates"]))
    np.testing.assert_allclose(actual, reference, rtol=0.0, atol=2.0e-15)

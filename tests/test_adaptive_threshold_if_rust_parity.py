# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Standalone Rust/Python adaptive-threshold parity

"""Execute the separately compilable Rust safety mirror on configured drive."""

from __future__ import annotations

from pathlib import Path
import subprocess

import numpy as np
import numpy.typing as npt

from sc_neurocore.accel.adaptive_threshold_if import simulate_python

_ROOT = Path(__file__).resolve().parents[1]
_PARAMETERS = (-63.5, -52.5, -68.0, -67.0, -49.0, 4.5, 8.0, 42.0, 0.05)


def _drive(steps: int) -> npt.NDArray[np.float64]:
    index = np.arange(steps, dtype=np.float64)
    return 22.0 + 6.0 * np.sin(index * 0.037) + 1.5 * np.cos(index * 0.011)


def test_standalone_rust_configured_trace_matches_python(tmp_path: Path) -> None:
    """Compare all states and sampled events with identical float64 inputs."""
    source = _ROOT / "src/sc_neurocore/accel/rust/safety/adaptive_threshold_if.rs"
    drive = _drive(128)
    literals = ",\n        ".join(f"{float(value):.17e}_f64" for value in drive)
    program = tmp_path / "configured_trace.rs"
    binary = tmp_path / "configured_trace"
    program.write_text(
        f'''include!(r#"{source}"#);

fn main() {{
    let drive = [
        {literals}
    ];
    let mut state = AdaptiveThresholdIFNeuron {{
        v: -63.5,
        theta: -52.5,
        v_rest: -68.0,
        v_reset: -67.0,
        theta_rest: -49.0,
        delta_theta: 4.5,
        tau_m: 8.0,
        tau_theta: 42.0,
        dt: 0.05,
    }};
    for current in drive {{
        let spike = state.step(current).expect("valid configured step");
        println!("{{:.17e}} {{:.17e}} {{}}", state.v, state.theta, spike);
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
        [[float(token) for token in line.split()] for line in completed.stdout.splitlines()],
        dtype=np.float64,
    )
    expected = simulate_python(*_PARAMETERS, drive)
    reference = np.column_stack((expected["v"], expected["theta"], expected["spikes"]))
    np.testing.assert_allclose(actual, reference, rtol=0.0, atol=2.0e-15)

# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Standalone Rust/Python alpha-synapse parity

"""Execute the separately compilable Rust safety mirror on configured drive."""

from __future__ import annotations

from pathlib import Path
import subprocess

import numpy as np
import numpy.typing as npt

from sc_neurocore.accel.alpha import simulate_python

_ROOT = Path(__file__).resolve().parents[1]
_PARAMETERS = (0.15, 0.08, 0.05, 0.04, 0.03, -0.5, 1.2, 16.0, 4.0, 9.0, 0.5)


def _drive(steps: int) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    index = np.arange(steps, dtype=np.float64)
    return 2.0 + 0.8 * np.sin(index * 0.037), 0.7 + 0.3 * np.cos(index * 0.021)


def test_standalone_rust_configured_trace_matches_python(tmp_path: Path) -> None:
    """Compare all states and sampled events with identical float64 inputs."""
    source = _ROOT / "src/sc_neurocore/accel/rust/safety/alpha.rs"
    exc, inh = _drive(128)
    exc_literals = ",\n        ".join(f"{float(value):.17e}_f64" for value in exc)
    inh_literals = ",\n        ".join(f"{float(value):.17e}_f64" for value in inh)
    program = tmp_path / "configured_trace.rs"
    binary = tmp_path / "configured_trace"
    program.write_text(
        f'''include!(r#"{source}"#);

fn main() {{
    let exc = [
        {exc_literals}
    ];
    let inh = [
        {inh_literals}
    ];
    let mut state = AlphaNeuron {{
        v: 0.15,
        a_exc: 0.08,
        i_exc: 0.05,
        a_inh: 0.04,
        i_inh: 0.03,
        v_rest: -0.5,
        v_threshold: 1.2,
        tau_v: 16.0,
        tau_exc: 4.0,
        tau_inh: 9.0,
        dt: 0.5,
    }};
    for index in 0..128 {{
        let spike = state.step(exc[index], inh[index]).expect("valid configured step");
        println!("{{:.17e}} {{:.17e}} {{:.17e}} {{}}", state.v, state.a_exc, state.i_exc, spike);
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
    expected = simulate_python(*_PARAMETERS, exc, inh)
    reference = np.column_stack(
        (expected["v"], expected["a_exc"], expected["i_exc"], expected["spikes"])
    )
    np.testing.assert_allclose(actual, reference, rtol=0.0, atol=2.0e-15)

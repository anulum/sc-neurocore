# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Executed Ih backend and custody parity

"""Execute every declared Ih backend and bind public claims to live sources."""

from __future__ import annotations

import math
from pathlib import Path
import shutil
import subprocess

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility path.
    import tomli as tomllib
from typing import Protocol, cast

import numpy as np
import pytest

from sc_neurocore.neurons.models.ih_neuron import IhNeuron

_ROOT = Path(__file__).resolve().parents[1]
_DRIVE = tuple(2.0 + 3.0 * math.sin(index * 0.17) for index in range(64))


class _RustIh(Protocol):
    def step(self, current: float) -> int: ...

    def get_state(self) -> dict[str, float]: ...


def _python_trace() -> np.ndarray:
    neuron = IhNeuron()
    rows = []
    for current in _DRIVE:
        event = neuron.step(current)
        rows.append((neuron.v, neuron.h, neuron.n, neuron.r, event))
    return np.asarray(rows, dtype=np.float64)


def _parse_trace(stdout: str) -> np.ndarray:
    return np.asarray(
        [[float(token) for token in line.split()] for line in stdout.splitlines()],
        dtype=np.float64,
    )


def test_production_rust_binding_matches_complete_python_state() -> None:
    engine = pytest.importorskip("sc_neurocore_engine.sc_neurocore_engine")
    rust = cast(_RustIh, engine.IhNeuron())
    actual = []
    for current in _DRIVE:
        event = rust.step(current)
        state = rust.get_state()
        actual.append((state["v"], state["h"], state["n"], state["r"], event))
    np.testing.assert_allclose(actual, _python_trace(), rtol=0.0, atol=1.0e-12)

    before = rust.get_state()
    with pytest.raises(ValueError, match="current"):
        rust.step(math.nan)
    assert rust.get_state() == before


def test_standalone_safety_rust_matches_complete_python_state(tmp_path: Path) -> None:
    rustc = shutil.which("rustc")
    if rustc is None:
        pytest.skip("rustc is not installed")
    source = _ROOT / "src/sc_neurocore/accel/rust/safety/ih_neuron.rs"
    literals = ", ".join(f"{current:.17e}_f64" for current in _DRIVE)
    program = tmp_path / "ih_trace.rs"
    binary = tmp_path / "ih_trace"
    program.write_text(
        f'''#[path = r#"{source}"#]
mod ih;

use ih::IhNeuron;

fn main() {{
    let mut state = IhNeuron::new();
    for current in [{literals}] {{
        let event = state.step(current).expect("finite configured drive");
        println!("{{:.17e}} {{:.17e}} {{:.17e}} {{:.17e}} {{}}", state.v, state.h, state.n, state.r, event);
    }}
}}
''',
        encoding="utf-8",
    )
    subprocess.run(
        [rustc, "--edition", "2021", "-O", str(program), "-o", str(binary)],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    completed = subprocess.run(
        [str(binary)], check=True, capture_output=True, text=True, timeout=30
    )
    np.testing.assert_allclose(
        _parse_trace(completed.stdout), _python_trace(), rtol=0.0, atol=1.0e-12
    )


def test_go_backend_matches_complete_python_state(tmp_path: Path) -> None:
    go = shutil.which("go")
    if go is None:
        pytest.skip("Go is not installed")
    literals = ", ".join(f"{current:.17e}" for current in _DRIVE)
    program = tmp_path / "ih_trace.go"
    program.write_text(
        f"""package main

import (
    "fmt"
    services "github.com/anulum/sc-neurocore/accel/services"
)

func main() {{
    state := services.NewIhNeuron()
    for _, current := range []float64{{{literals}}} {{
        event, err := state.TryStep(current)
        if err != nil {{ panic(err) }}
        fmt.Printf("%.17e %.17e %.17e %.17e %d\\n", state.V, state.H, state.N, state.R, event)
    }}
}}
""",
        encoding="utf-8",
    )
    completed = subprocess.run(
        [go, "run", str(program)],
        cwd=_ROOT / "src/sc_neurocore/accel/go",
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    np.testing.assert_allclose(
        _parse_trace(completed.stdout), _python_trace(), rtol=0.0, atol=1.0e-12
    )


def test_julia_backend_matches_complete_python_state() -> None:
    julia = shutil.which("julia")
    if julia is None:
        pytest.skip("Julia is not installed")
    source = _ROOT / "src/sc_neurocore/accel/julia/neurons/ih_neuron.jl"
    literals = ", ".join(f"{current:.17e}" for current in _DRIVE)
    program = f'''
include(raw"{source}")
using .IhNeuronAccel
state = IhNeuronState()
for current in [{literals}]
    event = step!(state, current)
    println(state.v, " ", state.h, " ", state.n, " ", state.r, " ", event)
end
before = (state.v, state.h, state.n, state.r)
try
    step!(state, NaN)
    error("NaN drive was accepted")
catch error
    error isa ArgumentError || rethrow()
end
(state.v, state.h, state.n, state.r) == before || error("invalid drive mutated state")
'''
    completed = subprocess.run(
        [julia, "--startup-file=no", "-e", program],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    np.testing.assert_allclose(
        _parse_trace(completed.stdout), _python_trace(), rtol=0.0, atol=1.0e-12
    )


def test_descriptor_and_public_page_report_only_proven_backends() -> None:
    descriptor = tomllib.loads(
        (_ROOT / "src/sc_neurocore/neurons/model_descriptors/IhNeuron.toml").read_text(
            encoding="utf-8"
        )
    )
    assert set(descriptor["backends"]) == {"python", "rust", "go", "julia"}
    assert all(config["status"] == "implemented" for config in descriptor["backends"].values())

    page = (_ROOT / "docs/api/models/ih_neuron.md").read_text(encoding="utf-8")
    assert "**Reference:** Robinson & Bhatt" not in page
    assert "publication-exact" in page
    assert "engine/src/neurons/channels/ih.rs" in page
    assert "Mojo" in page and "not implemented" in page

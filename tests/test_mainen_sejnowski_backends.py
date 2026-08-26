# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Executed Mainen-Sejnowski backend and custody parity

"""Execute every declared Mainen-Sejnowski backend against the reference."""

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

from sc_neurocore.neurons.models.mainen_sejnowski import MainenSejnowskiNeuron

_ROOT = Path(__file__).resolve().parents[1]
_DRIVE = tuple(4.0 + 3.0 * math.sin(index * 0.17) for index in range(64))


class _RustMainen(Protocol):
    def step(self, current: float) -> int: ...

    def get_state(self) -> dict[str, float]: ...


def _python_trace() -> np.ndarray:
    neuron = MainenSejnowskiNeuron()
    rows = []
    for current in _DRIVE:
        event = neuron.step(current)
        rows.append((neuron.vs, neuron.va, neuron.m, neuron.h, neuron.n, event))
    return np.asarray(rows, dtype=np.float64)


def _parse_trace(stdout: str) -> np.ndarray:
    return np.asarray(
        [[float(token) for token in line.split()] for line in stdout.splitlines()],
        dtype=np.float64,
    )


def test_production_rust_binding_matches_complete_python_state() -> None:
    engine = pytest.importorskip("sc_neurocore_engine.sc_neurocore_engine")
    rust = cast(_RustMainen, engine.MainenSejnowskiNeuron())
    actual = []
    for current in _DRIVE:
        event = rust.step(current)
        state = rust.get_state()
        actual.append((state["vs"], state["va"], state["m"], state["h"], state["n"], event))
    np.testing.assert_allclose(actual, _python_trace(), rtol=0.0, atol=1.0e-12)

    before = rust.get_state()
    with pytest.raises(ValueError, match="current"):
        rust.step(math.nan)
    with pytest.raises(ValueError, match="current"):
        rust.step(math.inf)
    assert rust.get_state() == before


def test_production_rust_binding_preserves_legacy_sequential_configuration() -> None:
    engine = pytest.importorskip("sc_neurocore_engine.sc_neurocore_engine")
    legacy = cast(_RustMainen, engine.MainenSejnowskiNeuron.legacy_sequential())
    assert legacy.step(10.0) == 0
    state = legacy.get_state()
    assert state["vs"] == pytest.approx(-32.668480035293555, abs=1e-12)
    assert state["h"] == pytest.approx(0.6581322365919791, abs=1e-12)
    assert state["n"] == pytest.approx(0.3981986218090308, abs=1e-12)


def test_standalone_safety_rust_matches_complete_python_state(tmp_path: Path) -> None:
    rustc = shutil.which("rustc")
    if rustc is None:
        pytest.skip("rustc is not installed")
    source = _ROOT / "src/sc_neurocore/accel/rust/safety/mainen_sejnowski.rs"
    literals = ", ".join(f"{current:.17e}_f64" for current in _DRIVE)
    program = tmp_path / "mainen_trace.rs"
    binary = tmp_path / "mainen_trace"
    program.write_text(
        f'''#[path = r#"{source}"#]
mod mainen;

use mainen::MainenSejnowskiNeuron;

fn main() {{
    let mut state = MainenSejnowskiNeuron::new();
    for current in [{literals}] {{
        let event = state.step(current).expect("finite configured drive");
        println!("{{:.17e}} {{:.17e}} {{:.17e}} {{:.17e}} {{:.17e}} {{}}", state.vs, state.va, state.m, state.h, state.n, event);
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
    program = tmp_path / "mainen_trace.go"
    program.write_text(
        f"""package main

import (
    "fmt"
    services "github.com/anulum/sc-neurocore/accel/services"
)

func main() {{
    state := services.NewMainenSejnowskiNeuron()
    for _, current := range []float64{{{literals}}} {{
        event, err := state.TryStep(current)
        if err != nil {{ panic(err) }}
        fmt.Printf("%.17e %.17e %.17e %.17e %.17e %d\\n", state.Vs, state.Va, state.M, state.H, state.N, event)
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
    source = _ROOT / "src/sc_neurocore/accel/julia/neurons/mainen_sejnowski.jl"
    literals = ", ".join(f"{current:.17e}" for current in _DRIVE)
    program = f'''
include(raw"{source}")
using .MainenSejnowskiAccel
state = MainenSejnowskiNeuronState()
for current in [{literals}]
    event = step!(state, current)
    println(state.vs, " ", state.va, " ", state.m, " ", state.h, " ", state.n, " ", event)
end
before = (state.vs, state.va, state.m, state.h, state.n)
for bad in (NaN, Inf, -Inf)
    try
        step!(state, bad)
        error("non-finite drive was accepted")
    catch error
        error isa ArgumentError || rethrow()
    end
end
(state.vs, state.va, state.m, state.h, state.n) == before || error("invalid drive mutated state")
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
        (_ROOT / "src/sc_neurocore/neurons/model_descriptors/MainenSejnowskiNeuron.toml").read_text(
            encoding="utf-8"
        )
    )
    assert set(descriptor["backends"]) == {"python", "rust", "go", "julia"}
    assert all(config["status"] == "implemented" for config in descriptor["backends"].values())
    assert descriptor["provenance"]["doi"] == "10.1038/382363a0"

    page = (_ROOT / "docs/api/models/mainen_sejnowski.md").read_text(encoding="utf-8")
    assert "382:363" in page
    assert "Mojo" in page and "not implemented" in page
    assert "legacy" in page

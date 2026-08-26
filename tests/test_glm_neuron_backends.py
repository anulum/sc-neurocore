# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Executed GLM backend and custody parity

"""Execute every declared GLM backend under the explicit-uniform contract."""

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

from sc_neurocore.neurons.models.glm_neuron import GLMNeuron

_ROOT = Path(__file__).resolve().parents[1]
_STEPS = 64
_DRIVE = tuple(4.0 + 3.0 * math.sin(index * 0.17) for index in range(_STEPS))
_UNIFORM = tuple(((index * 37 + 11) % 97) / 97.0 for index in range(_STEPS))


class _RustGLM(Protocol):
    def step(self, stimulus: float, uniform: float | None = None) -> int: ...

    def get_state(self) -> dict[str, object]: ...


def _python_trace() -> np.ndarray:
    neuron = GLMNeuron(seed=0)
    rows = []
    for current, sample in zip(_DRIVE, _UNIFORM, strict=True):
        event = neuron.step(current, uniform=sample)
        rows.append(
            (
                float(neuron._stim_buf[0]),
                float(neuron._spike_buf[0]),
                float(np.sum(neuron._stim_buf)),
                float(np.sum(neuron._spike_buf)),
                event,
            )
        )
    return np.asarray(rows, dtype=np.float64)


def _parse_trace(stdout: str) -> np.ndarray:
    return np.asarray(
        [[float(token) for token in line.split()] for line in stdout.splitlines()],
        dtype=np.float64,
    )


def test_production_rust_binding_matches_complete_python_state() -> None:
    engine = pytest.importorskip("sc_neurocore_engine.sc_neurocore_engine")
    rust = cast(_RustGLM, engine.GLMNeuron())
    actual = []
    for current, sample in zip(_DRIVE, _UNIFORM, strict=True):
        event = rust.step(current, sample)
        state = rust.get_state()
        stim = np.asarray(state["stim_buf"], dtype=np.float64)
        spikes = np.asarray(state["spike_buf"], dtype=np.float64)
        actual.append((stim[0], spikes[0], float(np.sum(stim)), float(np.sum(spikes)), event))
    np.testing.assert_allclose(actual, _python_trace(), rtol=0.0, atol=1.0e-12)

    before = rust.get_state()
    with pytest.raises(ValueError, match="stimulus"):
        rust.step(math.nan, 0.5)
    with pytest.raises(ValueError, match="uniform"):
        rust.step(1.0, 1.0)
    after = rust.get_state()
    assert np.array_equal(np.asarray(before["stim_buf"]), np.asarray(after["stim_buf"]))
    assert np.array_equal(np.asarray(before["spike_buf"]), np.asarray(after["spike_buf"]))


def test_production_rust_binding_preserves_legacy_constant_filter_configuration() -> None:
    engine = pytest.importorskip("sc_neurocore_engine.sc_neurocore_engine")
    legacy = engine.GLMNeuron.legacy_constant_filters(5, 10, 42)
    state = legacy.get_state()
    np.testing.assert_array_equal(np.asarray(state["k"]), np.full(5, 0.1))
    np.testing.assert_array_equal(np.asarray(state["h"]), np.full(10, -0.5))


def test_standalone_safety_rust_matches_complete_python_state(tmp_path: Path) -> None:
    rustc = shutil.which("rustc")
    if rustc is None:
        pytest.skip("rustc is not installed")
    source = _ROOT / "src/sc_neurocore/accel/rust/safety/glm_neuron.rs"
    drive = ", ".join(f"{current:.17e}_f64" for current in _DRIVE)
    uniforms = ", ".join(f"{sample:.17e}_f64" for sample in _UNIFORM)
    program = tmp_path / "glm_trace.rs"
    binary = tmp_path / "glm_trace"
    program.write_text(
        f'''#[path = r#"{source}"#]
mod glm;

use glm::GLMNeuron;

fn main() {{
    let mut state = GLMNeuron::new(10, 20);
    let drive = [{drive}];
    let uniforms = [{uniforms}];
    for (current, sample) in drive.iter().zip(uniforms.iter()) {{
        let event = state.step(*current, *sample).expect("finite configured drive");
        let stim_sum: f64 = state.stim_buf.iter().sum();
        let spike_sum: f64 = state.spike_buf.iter().sum();
        println!("{{:.17e}} {{:.17e}} {{:.17e}} {{:.17e}} {{}}", state.stim_buf[0], state.spike_buf[0], stim_sum, spike_sum, event);
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
    drive = ", ".join(f"{current:.17e}" for current in _DRIVE)
    uniforms = ", ".join(f"{sample:.17e}" for sample in _UNIFORM)
    program = tmp_path / "glm_trace.go"
    program.write_text(
        f"""package main

import (
    "fmt"
    services "github.com/anulum/sc-neurocore/accel/services"
)

func main() {{
    state := services.NewGLMNeuron(10, 20)
    drive := []float64{{{drive}}}
    uniforms := []float64{{{uniforms}}}
    for i := range drive {{
        event, err := state.TryStep(drive[i], uniforms[i])
        if err != nil {{ panic(err) }}
        stimSum, spikeSum := 0.0, 0.0
        for _, value := range state.StimBuf {{ stimSum += value }}
        for _, value := range state.SpikeBuf {{ spikeSum += value }}
        fmt.Printf("%.17e %.17e %.17e %.17e %d\\n", state.StimBuf[0], state.SpikeBuf[0], stimSum, spikeSum, event)
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
    source = _ROOT / "src/sc_neurocore/accel/julia/neurons/glm_neuron.jl"
    drive = ", ".join(f"{current:.17e}" for current in _DRIVE)
    uniforms = ", ".join(f"{sample:.17e}" for sample in _UNIFORM)
    program = f'''
include(raw"{source}")
using .GlmNeuronAccel
state = GLMNeuronState()
drive = [{drive}]
uniforms = [{uniforms}]
for i in eachindex(drive)
    event = step!(state, drive[i], uniforms[i])
    println(state.stim_buf[1], " ", state.spike_buf[1], " ", sum(state.stim_buf), " ", sum(state.spike_buf), " ", event)
end
before = (copy(state.stim_buf), copy(state.spike_buf))
for bad in (NaN, Inf, -Inf)
    try
        step!(state, bad, 0.5)
        error("non-finite stimulus was accepted")
    catch error
        error isa ArgumentError || rethrow()
    end
end
try
    step!(state, 1.0, 1.0)
    error("uniform >= 1 was accepted")
catch error
    error isa ArgumentError || rethrow()
end
(state.stim_buf == before[1] && state.spike_buf == before[2]) || error("invalid input mutated state")
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
        (_ROOT / "src/sc_neurocore/neurons/model_descriptors/GLMNeuron.toml").read_text(
            encoding="utf-8"
        )
    )
    assert set(descriptor["backends"]) == {"python", "rust", "go", "julia"}
    assert all(config["status"] == "implemented" for config in descriptor["backends"].values())
    assert descriptor["provenance"]["doi"] == "10.1038/nature07140"
    assert descriptor["integration"]["dt"] == 1.0

    page = (_ROOT / "docs/api/models/glm_neuron.md").read_text(encoding="utf-8")
    assert "995" in page
    assert "Mojo" in page and "not implemented" in page

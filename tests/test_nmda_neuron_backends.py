# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — independently executed NMDA native parity

from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
from typing import Protocol, cast

import numpy as np
import pytest

from sc_neurocore.accel.mojo.isa_baseline import pin_isa
from sc_neurocore.neurons.models import NMDANeuron, SCWBNMDAMagnesiumBlockNeuron
from tests.engine_requirement import require_engine

require_engine()
import sc_neurocore_engine as engine

_ROOT = Path(__file__).resolve().parents[1]
_SOURCE = np.array([-69.9700375, 0.0, 0.0, 0.0, 0.0])
_SC = np.array([-63.15566378039578, 0.6480311943997441, 0.237221887163776, 0.025])


class _EngineNeuron(Protocol):
    def step(self, current: float) -> int: ...

    def get_state(self) -> dict[str, float]: ...


def test_python_dual_identity_anchors() -> None:
    source = NMDANeuron()
    retained = SCWBNMDAMagnesiumBlockNeuron()
    assert source.step(0.3) == 0
    assert retained.step(5.0) == 0
    np.testing.assert_allclose(
        [source.v, source.x_nmda, source.s_nmda, source.ca, source.refractory_remaining],
        _SOURCE,
        rtol=0,
        atol=2e-14,
    )
    np.testing.assert_allclose(
        [retained.v, retained.h, retained.n, retained.s_nmda], _SC, rtol=0, atol=2e-14
    )


@pytest.mark.parametrize(
    ("constructor", "current", "keys", "expected"),
    [
        (
            engine.NMDANeuron,
            0.3,
            ("v", "x_nmda", "s_nmda", "ca", "refractory_remaining"),
            _SOURCE,
        ),
        (
            engine.SCWBNMDAMagnesiumBlockNeuron,
            5.0,
            ("v", "h", "n", "s_nmda"),
            _SC,
        ),
    ],
)
def test_production_rust_binding_dual_identity_parity(
    constructor: type[object], current: float, keys: tuple[str, ...], expected: np.ndarray
) -> None:
    neuron = cast(_EngineNeuron, constructor())
    assert neuron.step(current) == 0
    state = neuron.get_state()
    np.testing.assert_allclose([state[key] for key in keys], expected, rtol=0, atol=2e-12)
    before = neuron.get_state()
    with pytest.raises(ValueError, match="current"):
        neuron.step(float("nan"))
    assert neuron.get_state() == before


@pytest.mark.skipif(shutil.which("rustc") is None, reason="rustc is unavailable")
def test_standalone_safety_rust_dual_identity_parity(tmp_path: Path) -> None:
    source = _ROOT / "src/sc_neurocore/accel/rust/safety/nmda_neuron.rs"
    retained = _ROOT / "src/sc_neurocore/accel/rust/safety/sc_wb_nmda_magnesium_block.rs"
    program = tmp_path / "nmda_anchor.rs"
    binary = tmp_path / "nmda_anchor"
    program.write_text(
        f'''#[path = r#"{source}"#] mod source;
#[path = r#"{retained}"#] mod retained;
fn main() {{
    let mut a = source::NMDANeuron::new();
    let ae = a.step(0.3).expect("valid source step");
    println!("{{}} {{:.17e}} {{:.17e}} {{:.17e}} {{:.17e}} {{:.17e}}", ae, a.v, a.x_nmda, a.s_nmda, a.ca, a.refractory_remaining);
    let mut b = retained::SCWBNMDAMagnesiumBlockNeuron::new();
    let be = b.step(5.0).expect("valid retained step");
    println!("{{}} {{:.17e}} {{:.17e}} {{:.17e}} {{:.17e}}", be, b.v, b.h, b.n, b.s_nmda);
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
    lines = subprocess.run(
        [str(binary)], check=True, capture_output=True, text=True, timeout=30
    ).stdout.splitlines()
    source_values = [float(value) for value in lines[0].split()]
    retained_values = [float(value) for value in lines[1].split()]
    assert source_values[0] == retained_values[0] == 0.0
    np.testing.assert_allclose(source_values[1:], _SOURCE, rtol=0, atol=2e-12)
    np.testing.assert_allclose(retained_values[1:], _SC, rtol=0, atol=2e-12)


@pytest.mark.skipif(shutil.which("go") is None, reason="Go is unavailable")
def test_go_dual_identity_parity(tmp_path: Path) -> None:
    program = tmp_path / "nmda_anchor.go"
    program.write_text(
        """package main
import (
    "fmt"
    services "github.com/anulum/sc-neurocore/accel/services"
)
func main() {
    a := services.NewNMDANeuron()
    ae, err := a.TryStep(0.3); if err != nil { panic(err) }
    fmt.Printf("%d %.17e %.17e %.17e %.17e %.17e\\n", ae, a.V, a.XNmda, a.SNmda, a.Ca, a.RefractoryRemaining)
    b := services.NewSCWBNMDAMagnesiumBlockNeuron()
    be, err := b.TryStep(5.0); if err != nil { panic(err) }
    fmt.Printf("%d %.17e %.17e %.17e %.17e\\n", be, b.V, b.H, b.N, b.SNmda)
}
""",
        encoding="utf-8",
    )
    lines = subprocess.run(
        ["go", "run", str(program)],
        cwd=_ROOT / "src/sc_neurocore/accel/go",
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    ).stdout.splitlines()
    source_values = [float(value) for value in lines[0].split()]
    retained_values = [float(value) for value in lines[1].split()]
    assert source_values[0] == retained_values[0] == 0.0
    np.testing.assert_allclose(source_values[1:], _SOURCE, rtol=0, atol=2e-12)
    np.testing.assert_allclose(retained_values[1:], _SC, rtol=0, atol=2e-12)


@pytest.mark.skipif(shutil.which("julia") is None, reason="Julia is unavailable")
def test_julia_dual_identity_parity() -> None:
    source = _ROOT / "src/sc_neurocore/accel/julia/neurons/nmda_neuron.jl"
    retained = _ROOT / "src/sc_neurocore/accel/julia/neurons/sc_wb_nmda_magnesium_block.jl"
    expression = f'''
include(raw"{source}"); using .NmdaNeuronAccel
a=NMDANeuronState(); ae=step!(a,0.3)
println("[",ae,",",a.v,",",a.x_nmda,",",a.s_nmda,",",a.ca,",",a.refractory_remaining,"]")
include(raw"{retained}"); using .SCWBNMDAMagnesiumBlockAccel
b=SCWBNMDAMagnesiumBlockNeuronState(); be=SCWBNMDAMagnesiumBlockAccel.step!(b,5.0)
println("[",be,",",b.v,",",b.h,",",b.n,",",b.s_nmda,"]")
'''
    lines = subprocess.run(
        ["julia", "--startup-file=no", "-e", expression],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    ).stdout.splitlines()
    source_values = np.asarray(json.loads(lines[0]), dtype=np.float64)
    retained_values = np.asarray(json.loads(lines[1]), dtype=np.float64)
    assert source_values[0] == retained_values[0] == 0.0
    np.testing.assert_allclose(source_values[1:], _SOURCE, rtol=0, atol=2e-12)
    np.testing.assert_allclose(retained_values[1:], _SC, rtol=0, atol=2e-12)


@pytest.mark.skipif(shutil.which("mojo") is None, reason="Mojo is unavailable")
@pytest.mark.parametrize(
    ("filename", "expected"),
    [("nmda_neuron.mojo", _SOURCE), ("sc_wb_nmda_magnesium_block.mojo", _SC)],
)
def test_mojo_dual_identity_parity(tmp_path: Path, filename: str, expected: np.ndarray) -> None:
    kernel = _ROOT / "src/sc_neurocore/accel/mojo/kernels" / filename
    binary = tmp_path / kernel.stem
    command = [
        "mojo",
        "build",
        "--disable-warnings",
        "-Xlinker",
        "-lm",
        str(kernel),
        "-o",
        str(binary),
    ]
    subprocess.run(pin_isa(command), check=True, timeout=120)
    values = [
        float(value)
        for value in subprocess.run(
            [str(binary)], check=True, capture_output=True, text=True, timeout=30
        )
        .stdout.splitlines()[0]
        .split()
    ]
    assert values[0] == 0.0
    np.testing.assert_allclose(values[1:], expected, rtol=0, atol=2e-12)


def test_declared_backends_and_public_boundary_are_exact() -> None:
    source = (_ROOT / "src/sc_neurocore/neurons/model_descriptors/NMDANeuron.toml").read_text(
        encoding="utf-8"
    )
    assert all(
        f"[backends.{backend}]" in source
        for backend in ("python", "rust_engine", "rust_safety", "go", "julia", "mojo")
    )
    assert "[silicon]" in source
    assert "cosim_validated = true" in source
    assert "binary64 formal equivalence remain open" in source

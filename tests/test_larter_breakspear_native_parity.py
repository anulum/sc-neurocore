# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — independently executed Larter-Breakspear native parity

from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest

from sc_neurocore.accel.mojo.isa_baseline import pin_isa
from sc_neurocore.neurons.models.larter_breakspear import LarterBreakspearNeuron
from sc_neurocore.neurons.models.sc_decoupled_adaptation_ion_mass import (
    SCDecoupledAdaptationIonMassNeuron,
)

_ROOT = Path(__file__).resolve().parents[1]
_SOURCE = np.array([0.10851023903311285, 0.10395561081677293, 0.10024082963711924])
_SC = np.array([-0.4987593419078305, 0.0002412377194581311, 6.202934920494744e-7])


def test_criterion_series_keep_source_and_sc_identities_separate() -> None:
    benchmark = (_ROOT / "engine/benches/full_bench.rs").read_text()

    assert '"larter_breakspear_100k_steps"' not in benchmark
    assert '"larter_breakspear_2003_rk4_100k_steps"' in benchmark
    assert '"sc_decoupled_adaptation_ion_mass_100k_steps"' in benchmark


def test_python_dual_identity_anchors() -> None:
    source = LarterBreakspearNeuron()
    retained = SCDecoupledAdaptationIonMassNeuron()
    source.step()
    retained.step()
    np.testing.assert_allclose([source.v, source.w, source.z], _SOURCE, rtol=0, atol=2e-14)
    np.testing.assert_allclose([retained.v, retained.w, retained.z], _SC, rtol=0, atol=2e-14)


@pytest.mark.skipif(shutil.which("julia") is None, reason="Julia runtime unavailable")
@pytest.mark.parametrize(
    ("filename", "module", "constructor", "expected"),
    [
        ("larter_breakspear.jl", "LarterBreakspearAccel", "LarterBreakspearNeuronState", _SOURCE),
        (
            "sc_decoupled_adaptation_ion_mass.jl",
            "SCDecoupledAdaptationIonMassAccel",
            "SCDecoupledAdaptationIonMassNeuronState",
            _SC,
        ),
    ],
)
def test_julia_dual_identity_parity(
    filename: str, module: str, constructor: str, expected: np.ndarray
) -> None:
    kernel = _ROOT / "src/sc_neurocore/accel/julia/neurons" / filename
    expression = (
        f'include(raw"{kernel}"); using .{module}; s={constructor}(); step!(s); '
        'print("[",s.v,",",s.w,",",s.z,"]")'
    )
    values = json.loads(
        subprocess.run(
            ["julia", "--startup-file=no", "-e", expression],
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        ).stdout
    )
    np.testing.assert_allclose(values, expected, rtol=0, atol=2e-12)


@pytest.mark.skipif(shutil.which("mojo") is None, reason="Mojo runtime unavailable")
@pytest.mark.parametrize(
    ("filename", "expected"),
    [("larter_breakspear.mojo", _SOURCE), ("sc_decoupled_adaptation_ion_mass.mojo", _SC)],
)
def test_mojo_dual_identity_parity(tmp_path: Path, filename: str, expected: np.ndarray) -> None:
    kernel = _ROOT / "src/sc_neurocore/accel/mojo/kernels" / filename
    binary = tmp_path / kernel.stem
    command = pin_isa(
        [
            "mojo",
            "build",
            "--disable-warnings",
            "-Xlinker",
            "-lm",
            str(kernel),
            "-o",
            str(binary),
        ]
    )
    subprocess.run(command, check=True, timeout=60)
    values = [
        float(value)
        for value in subprocess.run(
            [str(binary)],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        .stdout.splitlines()[0]
        .split()
    ]
    np.testing.assert_allclose(values, expected, rtol=0, atol=2e-12)

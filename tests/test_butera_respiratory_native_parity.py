# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — independently executed Butera Model 1 native parity

from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest

from sc_neurocore.accel.mojo.isa_baseline import pin_isa
from sc_neurocore.neurons.models.butera_respiratory import ButeraRespiratoryNeuron
from sc_neurocore.neurons.models.sc_unit_capacitance_respiratory import (
    SCUnitCapacitanceRespiratoryNeuron,
)

_ROOT = Path(__file__).resolve().parents[1]
_ONE_STEP = np.array([-50.033785184435395, 0.009677458379988083, 0.5000008443290977])
_SC_ONE_STEP = np.array([-50.71307220472, 0.009634244179940929, 0.5000009881782027])


def test_python_source_one_step_fixture() -> None:
    model = ButeraRespiratoryNeuron()
    assert model.step(12.5) == 0
    np.testing.assert_allclose([model.v, model.n, model.h_nap], _ONE_STEP, rtol=0, atol=2e-14)


@pytest.mark.skipif(shutil.which("julia") is None, reason="Julia runtime unavailable")
def test_julia_source_mirror_matches_python() -> None:
    kernel = _ROOT / "src/sc_neurocore/accel/julia/neurons/butera_respiratory.jl"
    expression = (
        f'include(raw"{kernel}"); using .ButeraRespiratoryAccel; '
        "s=ButeraRespiratoryNeuronState(); event=step!(s,12.5); "
        'print("[",s.v,",",s.n,",",s.h_nap,",",event,"]")'
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
    np.testing.assert_allclose(values[:3], _ONE_STEP, rtol=0, atol=2e-12)
    assert values[3] == 0


@pytest.mark.skipif(shutil.which("mojo") is None, reason="Mojo runtime unavailable")
def test_mojo_source_mirror_matches_python(tmp_path: Path) -> None:
    kernel = _ROOT / "src/sc_neurocore/accel/mojo/kernels/butera_respiratory.mojo"
    binary = tmp_path / "butera_respiratory"
    subprocess.run(
        pin_isa(["mojo", "build", "--disable-warnings", str(kernel), "-o", str(binary)]),
        check=True,
        timeout=60,
    )
    lines = subprocess.run(
        [str(binary)], check=True, capture_output=True, text=True, timeout=30
    ).stdout.splitlines()
    values = [float(value) for value in lines[0].split()]
    np.testing.assert_allclose(values[:3], _ONE_STEP, rtol=0, atol=2e-12)
    assert values[3] == 0
    assert lines[1] == "butera_model1 spikes: 173"


def test_python_sc_identity_one_step_fixture() -> None:
    model = SCUnitCapacitanceRespiratoryNeuron()
    assert model.step(12.5) == 0
    np.testing.assert_allclose([model.v, model.n, model.h_nap], _SC_ONE_STEP, rtol=0, atol=2e-14)


@pytest.mark.skipif(shutil.which("julia") is None, reason="Julia runtime unavailable")
def test_julia_sc_identity_matches_python() -> None:
    kernel = _ROOT / "src/sc_neurocore/accel/julia/neurons/sc_unit_capacitance_respiratory.jl"
    expression = (
        f'include(raw"{kernel}"); const SC=SCUnitCapacitanceRespiratoryAccel; '
        "s=SC.SCUnitCapacitanceRespiratoryNeuronState(); event=SC.step!(s,12.5); "
        'print("[",s.v,",",s.n,",",s.h_nap,",",event,"]")'
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
    np.testing.assert_allclose(values[:3], _SC_ONE_STEP, rtol=0, atol=2e-12)
    assert values[3] == 0


@pytest.mark.skipif(shutil.which("mojo") is None, reason="Mojo runtime unavailable")
def test_mojo_sc_identity_matches_python(tmp_path: Path) -> None:
    kernel = _ROOT / "src/sc_neurocore/accel/mojo/kernels/sc_unit_capacitance_respiratory.mojo"
    binary = tmp_path / "sc_unit_capacitance_respiratory"
    subprocess.run(
        pin_isa(
            [
                "mojo",
                "build",
                "--disable-warnings",
                "-I",
                str(kernel.parent),
                str(kernel),
                "-o",
                str(binary),
            ]
        ),
        check=True,
        timeout=60,
    )
    lines = subprocess.run(
        [str(binary)], check=True, capture_output=True, text=True, timeout=30
    ).stdout.splitlines()
    values = [float(value) for value in lines[0].split()]
    np.testing.assert_allclose(values[:3], _SC_ONE_STEP, rtol=0, atol=2e-12)
    assert values[3] == 0
    assert int(lines[1].split()[-1]) in (4, 5)

# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — independently executed Bertram native parity

from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest

from sc_neurocore.neurons.models.bertram_phantom import BertramPhantomBurster
from sc_neurocore.neurons.models.sc_three_state_phantom import SCThreeStatePhantomBurster

_ROOT = Path(__file__).resolve().parents[1]
_EXPECTED = np.array(
    [-42.96246667898054, 0.030142733666228928, 0.0999512959452674, 0.4339985218163737]
)
_NATIVE_ATOL = 5e-13
_LONG_EXPECTED = np.array(
    [-27.635705868734412, 0.1969841125736782, 0.21414242579166587, 0.4341985645207124]
)
_SC_EXPECTED = np.array([-49.81865398074262, 0.10000426804815778, 0.09999950000126304])
_SC_LONG_EXPECTED = np.array([-58.49802324062697, 0.12687104203425778, 0.1048652930100963])


def test_python_reference_one_step_fixture() -> None:
    model = BertramPhantomBurster()
    assert model.step(0.0) == 0
    np.testing.assert_allclose(
        [model.v, model.n, model.s1, model.s2], _EXPECTED, rtol=0, atol=2e-14
    )


@pytest.mark.skipif(shutil.which("julia") is None, reason="Julia runtime unavailable")
def test_julia_source_mirror_matches_python() -> None:
    kernel = _ROOT / "src/sc_neurocore/accel/julia/neurons/bertram_phantom.jl"
    expression = (
        f'include(raw"{kernel}"); using .BertramPhantomAccel; '
        "s=BertramPhantomState(); event=step!(s,0.0); "
        'print("[",s.v,",",s.n,",",s.s1,",",s.s2,",",event,"]")'
    )
    output = subprocess.run(
        ["julia", "--startup-file=no", "-e", expression],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    ).stdout
    values = json.loads(output)
    np.testing.assert_allclose(values[:4], _EXPECTED, rtol=0, atol=_NATIVE_ATOL)
    assert values[4] == 0


@pytest.mark.skipif(shutil.which("mojo") is None, reason="Mojo runtime unavailable")
def test_mojo_source_mirror_matches_python(tmp_path: Path) -> None:
    kernel = _ROOT / "src/sc_neurocore/accel/mojo/kernels/bertram_phantom.mojo"
    binary = tmp_path / "bertram_phantom"
    subprocess.run(["mojo", "build", str(kernel), "-o", str(binary)], check=True, timeout=60)
    output = subprocess.run(
        [str(binary)], check=True, capture_output=True, text=True, timeout=20
    ).stdout.split()
    values = [float(value) for value in output]
    np.testing.assert_allclose(values[:4], _LONG_EXPECTED, rtol=0, atol=5e-9)
    assert values[4] == 18


def test_sc_python_compatibility_fixture() -> None:
    model = SCThreeStatePhantomBurster()
    assert model.step(0.0) == 0
    np.testing.assert_allclose([model.v, model.s1, model.s2], _SC_EXPECTED, rtol=0, atol=2e-14)


@pytest.mark.skipif(shutil.which("julia") is None, reason="Julia runtime unavailable")
def test_sc_julia_compatibility_mirror() -> None:
    kernel = _ROOT / "src/sc_neurocore/accel/julia/neurons/sc_three_state_phantom.jl"
    expression = (
        f'include(raw"{kernel}"); using .SCThreeStatePhantomAccel; '
        "s=SCThreeStatePhantomState(); event=step!(s,0.0); "
        'print("[",s.v,",",s.s1,",",s.s2,",",event,"]")'
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
    np.testing.assert_allclose(values[:3], _SC_EXPECTED, rtol=0, atol=2e-12)
    assert values[3] == 0


@pytest.mark.skipif(shutil.which("mojo") is None, reason="Mojo runtime unavailable")
def test_sc_mojo_compatibility_mirror(tmp_path: Path) -> None:
    kernel = _ROOT / "src/sc_neurocore/accel/mojo/kernels/sc_three_state_phantom.mojo"
    binary = tmp_path / "sc_three_state_phantom"
    subprocess.run(["mojo", "build", str(kernel), "-o", str(binary)], check=True, timeout=60)
    values = [
        float(value)
        for value in subprocess.run(
            [str(binary)], check=True, capture_output=True, text=True, timeout=20
        ).stdout.split()
    ]
    np.testing.assert_allclose(values[:3], _SC_LONG_EXPECTED, rtol=0, atol=2e-11)
    assert values[3] == 0

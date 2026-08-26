# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — independently executed Hill-Tononi native parity

from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest

from sc_neurocore.accel.mojo.isa_baseline import pin_isa
from sc_neurocore.neurons.models.hill_tononi import HillTononiNeuron

_ROOT = Path(__file__).resolve().parents[1]
_ONE_STEP = np.array(
    [
        -69.81228106951788,
        -51.0,
        0.0010000391293823398,
        0.2871847356365222,
        0.14517852005929924,
        0.03731808661830847,
    ]
)


def test_python_reference_one_step_fixture() -> None:
    model = HillTononiNeuron()
    assert model.step(12.0) == 0
    np.testing.assert_allclose(
        [model.v, model.theta, model.d_k, model.m_h, model.m_t, model.h_t],
        _ONE_STEP,
        rtol=0,
        atol=2e-14,
    )


@pytest.mark.skipif(shutil.which("julia") is None, reason="Julia runtime unavailable")
def test_julia_source_mirror_matches_python() -> None:
    kernel = _ROOT / "src/sc_neurocore/accel/julia/neurons/hill_tononi.jl"
    expression = (
        f'include(raw"{kernel}"); using .HillTononiAccel; '
        "s=HillTononiNeuronState(); event=step!(s,12.0); "
        'print("[",s.v,",",s.theta,",",s.d_k,",",s.m_h,",",s.m_t,",",s.h_t,",",event,"]")'
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
    np.testing.assert_allclose(values[:6], _ONE_STEP, rtol=0, atol=2e-12)
    assert values[6] == 0


@pytest.mark.skipif(shutil.which("mojo") is None, reason="Mojo runtime unavailable")
def test_mojo_source_mirror_matches_python(tmp_path: Path) -> None:
    kernel = _ROOT / "src/sc_neurocore/accel/mojo/kernels/hill_tononi.mojo"
    binary = tmp_path / "hill_tononi"
    subprocess.run(
        pin_isa(["mojo", "build", "--disable-warnings", str(kernel), "-o", str(binary)]),
        check=True,
        timeout=60,
    )
    lines = subprocess.run(
        [str(binary)], check=True, capture_output=True, text=True, timeout=30
    ).stdout.splitlines()
    values = [float(value) for value in lines[0].split()]
    np.testing.assert_allclose(values[:6], _ONE_STEP, rtol=0, atol=2e-12)
    assert values[6] == 0
    assert lines[1] == "hill_tononi spikes: 538"

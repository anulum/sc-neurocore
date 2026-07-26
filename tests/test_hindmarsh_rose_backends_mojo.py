# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Hindmarsh-Rose Mojo backend tests

"""ULP-bounded Mojo Hindmarsh–Rose parity contracts."""

from __future__ import annotations

import numpy as np
import pytest

from sc_neurocore.neurons.models.hindmarsh_rose import HindmarshRoseNeuron
from tests.hindmarsh_rose_backends_support import _STEP_TOL, _mojo, _run


@pytest.mark.skipif(not _mojo(), reason="Mojo Hindmarsh-Rose backend unavailable")
def test_mojo_per_step_within_tolerance() -> None:
    rng = np.random.default_rng(1)
    worst = 0.0
    for _ in range(5000):
        x = float(rng.uniform(-2.0, 2.0))
        y = float(rng.uniform(-12.0, 2.0))
        z = float(rng.uniform(0.0, 4.0))
        cur = float(rng.uniform(0.0, 4.0))
        ref, _rs, rx, ry, rz = HindmarshRoseNeuron(x=x, y=y, z=z)._simulate_python(1, cur)
        got, _gs, gx, gy, gz = HindmarshRoseNeuron(x=x, y=y, z=z)._simulate_mojo(1, cur)
        worst = max(worst, abs(ref[0] - got[0]), abs(rx - gx), abs(ry - gy), abs(rz - gz))
    assert worst <= _STEP_TOL, f"per-step Mojo gap {worst} exceeds {_STEP_TOL}"


@pytest.mark.skipif(not _mojo(), reason="Mojo Hindmarsh-Rose backend unavailable")
def test_mojo_trace_finite_and_short_horizon_tracks() -> None:
    # Over a short horizon (before chaos amplifies the FMA ULP) Mojo tracks the
    # reference closely; the full trace stays finite.
    ref, _rs, _rx, _ry, _rz = _run("python", current=3.0, n=8000)
    got, spikes, _gx, _gy, _gz = _run("mojo", current=3.0, n=8000)
    assert np.all(np.isfinite(got))
    np.testing.assert_allclose(got[:200], ref[:200], atol=1e-12, rtol=0.0)
    assert spikes == _rs

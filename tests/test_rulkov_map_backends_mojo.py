# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Focused Rulkov backend contracts

"""Focused cross-backend Rulkov map contracts."""

from .rulkov_map_backends_support import *


@pytest.mark.skipif(not _mojo(), reason="Mojo Rulkov backend unavailable")
@pytest.mark.parametrize("sigma", _REGIMES)
def test_mojo_trace_ulp_bounded(sigma: float) -> None:
    # The branch resets resynchronise the trajectory, so the whole-trace gap
    # stays at the per-step FMA level rather than diverging.
    ref, _ref_spikes, _rx, _ry = _run("python", sigma=sigma, n=2000)
    got, _spikes, _xf, _yf = _run("mojo", sigma=sigma, n=2000)
    np.testing.assert_allclose(got, ref, atol=1e-9, rtol=0.0)


@pytest.mark.skipif(not _mojo(), reason="Mojo Rulkov backend unavailable")
def test_mojo_per_step_within_tolerance() -> None:
    rng = np.random.default_rng(11)
    worst = 0.0
    for _ in range(5000):
        x = float(rng.uniform(-2.0, 0.5))
        y = float(rng.uniform(-4.0, -2.0))
        cur = float(rng.uniform(0.0, 2.0))
        ref, _rs, rx, ry = RulkovMapNeuron(x=x, y=y)._simulate_python(1, cur)
        got, _gs, gx, gy = RulkovMapNeuron(x=x, y=y)._simulate_mojo(1, cur)
        worst = max(worst, abs(ref[0] - got[0]), abs(rx - gx), abs(ry - gy))
    assert worst <= _STEP_TOL, f"per-step Mojo gap {worst} exceeds {_STEP_TOL}"


@pytest.mark.skipif(not _mojo(), reason="Mojo Rulkov backend unavailable")
def test_mojo_spike_count_matches() -> None:
    _ref, ref_spikes, _rx, _ry = _run("python", current=0.5, n=8000)
    _got, spikes, _xf, _yf = _run("mojo", current=0.5, n=8000)
    assert spikes == ref_spikes

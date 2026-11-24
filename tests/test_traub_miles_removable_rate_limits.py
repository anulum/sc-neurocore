# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Traub-Miles removable rate singularities

"""Traub-Miles rate functions at their three removable singularities.

``alpha_m``, ``beta_m`` and ``alpha_n`` have the forms ``a d / (1 - exp(-d / k))`` and
``a d / (exp(d / k) - 1)``, which are 0/0 at ``d = 0``. Their limits are ``a k``
(L'Hôpital): 0.32 * 4 = 1.28 at -54 mV, 0.28 * 5 = 1.40 at -27 mV and 0.032 * 5 = 0.16 at
-52 mV. Inside ``|d| <= 1e-6`` every implementation evaluates the first-order expansion
about that limit, and outside it the closed form, so the rates are continuous across the
window and unchanged away from it.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import shutil
import subprocess

import pytest

from sc_neurocore.neurons.models.traub_miles import TraubMilesNeuron

_ROOT = Path(__file__).resolve().parents[1]
_JULIA_KERNEL = _ROOT / "src/sc_neurocore/accel/julia/neurons/traub_miles.jl"
_WINDOW = 1e-6
# (rate index in _rates, singular voltage, analytic limit a * k)
_SINGULAR = ((0, -54.0, 1.28), (1, -27.0, 1.40), (4, -52.0, 0.16))


def _closed_form(index: int, v: float) -> float:
    """Evaluate the closed-form rate the models use outside the window."""
    if index == 0:
        d = v + 54.0
        return 0.32 * d / (1.0 - math.exp(-d / 4.0))
    if index == 1:
        d = v + 27.0
        return 0.28 * d / (math.exp(d / 5.0) - 1.0)
    d = v + 52.0
    return 0.032 * d / (1.0 - math.exp(-d / 5.0))


@pytest.mark.parametrize(("index", "v0", "limit"), _SINGULAR)
def test_rate_at_singular_voltage_is_the_analytic_limit(
    index: int, v0: float, limit: float
) -> None:
    """At d = 0 each rate equals a * k, the L'Hôpital limit."""
    assert TraubMilesNeuron._rates(v0)[index] == pytest.approx(limit, rel=0, abs=1e-15)


@pytest.mark.parametrize(("index", "v0", "limit"), _SINGULAR)
@pytest.mark.parametrize("offset", [1e-12, 1e-9, 1e-7, 5e-7, _WINDOW])
def test_rate_is_continuous_inside_the_window(
    index: int, v0: float, limit: float, offset: float
) -> None:
    """Inside the window the rate differs from the limit by its slope a / 2 times d."""
    for sign in (-1.0, 1.0):
        rate = TraubMilesNeuron._rates(v0 + sign * offset)[index]
        assert abs(rate - limit) <= 0.5 * offset + 1e-15


@pytest.mark.parametrize(("index", "v0", "limit"), _SINGULAR)
def test_window_edge_meets_the_closed_form(index: int, v0: float, limit: float) -> None:
    """Across the window edge the rate jumps by no more than the closed form's rounding.

    0.99 * 1e-6 is inside the window and 1.01 * 1e-6 outside it; the two rates differ by
    the slope times 2e-8 plus the cancellation error of the closed form there (under 1e-9).
    """
    for sign in (-1.0, 1.0):
        inside = TraubMilesNeuron._rates(v0 + sign * 0.99 * _WINDOW)[index]
        outside = TraubMilesNeuron._rates(v0 + sign * 1.01 * _WINDOW)[index]
        assert outside == _closed_form(index, v0 + sign * 1.01 * _WINDOW)
        assert abs(outside - inside) <= 0.5 * 0.02 * _WINDOW + 1e-9
        assert abs(inside - limit) <= 0.5 * _WINDOW + 1e-15


@pytest.mark.parametrize("v", [-80.0, -54.1, -53.9, -27.5, -26.5, -52.01, -51.99, 0.0, 30.0])
def test_rates_away_from_the_window_are_the_closed_form(v: float) -> None:
    """Away from the singularities the three rates are exactly the closed form."""
    rates = TraubMilesNeuron._rates(v)
    for index in (0, 1, 4):
        assert rates[index] == _closed_form(index, v)


def test_step_from_a_singular_voltage_uses_the_limit() -> None:
    """A step from exactly -54 mV matches one from 2e-6 mV away, outside the window.

    The second start uses the closed form; the first, the expansion about the limit. The
    2e-6 mV offset alone moves the sodium gate by about 2e-8 after one step; with the old
    constant 8.0 the gate moved about 5e-3 further.
    """
    at_singularity = TraubMilesNeuron(v=-54.0)
    outside = TraubMilesNeuron(v=-54.0 + 2.0 * _WINDOW)
    at_singularity.step(0.0)
    outside.step(0.0)
    assert at_singularity.m == pytest.approx(outside.m, rel=0, abs=1e-7)
    assert at_singularity.v == pytest.approx(outside.v, rel=0, abs=1e-5)


@pytest.mark.skipif(shutil.which("julia") is None, reason="Julia runtime unavailable")
def test_julia_source_mirror_matches_python_at_the_singularities() -> None:
    """The Julia lane's rates at and beside each singular voltage equal Python's bit for bit."""
    voltages = [v0 + offset for _, v0, _ in _SINGULAR for offset in (0.0, -5e-7, 5e-7, 3e-6)]
    expression = (
        f'include(raw"{_JULIA_KERNEL}"); using .TraubMilesAccel; '
        f"vs = {json.dumps(voltages)}; "
        'print("[", join([join(string.(collect(TraubMilesAccel.rates(v))), ",") for v in vs], ","), "]")'
    )
    output = subprocess.run(
        ["julia", "--startup-file=no", "-e", expression],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    ).stdout
    values = json.loads(output)
    expected = [rate for v in voltages for rate in TraubMilesNeuron._rates(v)]
    assert values == expected

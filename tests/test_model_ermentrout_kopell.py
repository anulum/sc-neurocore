# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — MPR scalar and atomic-batch contracts

"""Verify equation-(12) state, safety, and public simulation semantics."""

from __future__ import annotations

import math
from typing import cast

import numpy as np
import numpy.typing as npt
import pytest

from sc_neurocore.neurons.models.ermentrout_kopell_pop import (
    ErmentroutKopellPopulation,
)


def test_source_bound_defaults_and_legacy_identity_boundary() -> None:
    unit = ErmentroutKopellPopulation()
    assert (unit.r, unit.v) == (0.1, -2.0)
    assert (unit.tau, unit.delta, unit.eta_bar, unit.j, unit.dt) == (
        1.0,
        1.0,
        -5.0,
        15.0,
        0.01,
    )
    assert "Montbrió, Pazó, and Roxin" in (unit.__class__.__doc__ or "")
    assert "Ermentrout–Kopell theta model" in (unit.__class__.__doc__ or "")


def test_one_step_matches_equations_twelve_with_explicit_tau() -> None:
    unit = ErmentroutKopellPopulation(
        r=0.2,
        v=-1.5,
        tau=2.0,
        delta=0.7,
        eta_bar=-3.0,
        j=12.0,
        dt=0.005,
    )
    drive = 1.25
    old_r, old_v = unit.r, unit.v
    expected_r = old_r + unit.dt * (
        unit.delta / (math.pi * unit.tau**2) + 2.0 * old_r * old_v / unit.tau
    )
    expected_v = old_v + unit.dt * (
        (
            old_v**2
            + unit.eta_bar
            + drive
            + unit.j * unit.tau * old_r
            - (math.pi * unit.tau * old_r) ** 2
        )
        / unit.tau
    )
    returned = unit.step(drive)
    assert unit.r == expected_r
    assert unit.v == expected_v
    assert returned == expected_r


@pytest.mark.parametrize(
    "kwargs",
    (
        {"r": -0.1},
        {"tau": 0.0},
        {"tau": -1.0},
        {"delta": -0.1},
        {"dt": 0.0},
        {"v": math.nan},
        {"eta_bar": math.inf},
        {"j": -math.inf},
    ),
)
def test_invalid_configuration_is_rejected(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        ErmentroutKopellPopulation(**kwargs)


@pytest.mark.parametrize(
    ("name", "value"),
    (
        ("r", None),
        ("v", object()),
        ("tau", "not-a-number"),
    ),
)
def test_non_numeric_configuration_raises_value_error(
    name: str,
    value: object,
) -> None:
    kwargs: dict[str, float] = {name: cast(float, value)}
    with pytest.raises(ValueError, match=rf"^{name} must be numeric$"):
        ErmentroutKopellPopulation(**kwargs)


def test_rejected_input_and_candidate_are_atomic() -> None:
    unit = ErmentroutKopellPopulation()
    before = (unit.r, unit.v)
    with pytest.raises(ValueError, match="external input"):
        unit.step(math.nan)
    assert (unit.r, unit.v) == before

    with pytest.raises(ValueError, match="external input must be numeric"):
        unit.step(cast(float, None))
    assert (unit.r, unit.v) == before

    unstable = ErmentroutKopellPopulation(
        r=1.0,
        v=-100.0,
        delta=0.0,
        dt=0.01,
    )
    before_unstable = (unstable.r, unstable.v)
    with pytest.raises(FloatingPointError, match="negative"):
        unstable.step(0.0)
    assert (unstable.r, unstable.v) == before_unstable


def test_corrupted_state_is_rejected_without_mutating_peer_state() -> None:
    unit = ErmentroutKopellPopulation()
    unit.v = math.inf
    before = (unit.r, unit.v)
    with pytest.raises(ValueError, match="state"):
        unit.step(0.0)
    assert (unit.r, unit.v) == before


@pytest.mark.parametrize(
    ("name", "value", "message"),
    (
        ("r", object(), "numeric"),
        ("r", -0.1, "non-negative"),
    ),
)
def test_corrupted_state_type_or_rate_is_rejected_atomically(
    name: str,
    value: object,
    message: str,
) -> None:
    unit = ErmentroutKopellPopulation()
    setattr(unit, name, value)
    before = (unit.r, unit.v)
    with pytest.raises(ValueError, match=message):
        unit.step(0.0)
    assert (unit.r, unit.v) == before


@pytest.mark.parametrize(
    ("name", "value"),
    (
        ("tau", 0.0),
        ("delta", -0.1),
        ("eta_bar", math.inf),
        ("j", math.nan),
        ("dt", -0.01),
    ),
)
def test_corrupted_parameter_is_rejected_without_mutating_state(
    name: str,
    value: float,
) -> None:
    unit = ErmentroutKopellPopulation()
    setattr(unit, name, value)
    before = (unit.r, unit.v)
    with pytest.raises(ValueError):
        unit.step(0.0)
    assert (unit.r, unit.v) == before


def test_corrupted_parameter_type_is_rejected_atomically() -> None:
    unit = ErmentroutKopellPopulation()
    unit.tau = cast(float, object())
    before = (unit.r, unit.v)
    with pytest.raises(ValueError, match="parameters must be numeric"):
        unit.step(0.0)
    assert (unit.r, unit.v) == before


def test_nonfinite_candidate_is_rejected_atomically() -> None:
    unit = ErmentroutKopellPopulation(v=1.0e154, dt=2.0)
    before = (unit.r, unit.v)
    with pytest.raises(FloatingPointError, match="candidate must remain finite"):
        unit.step(0.0)
    assert (unit.r, unit.v) == before


def test_reset_preserves_parameters_and_restores_dynamic_state() -> None:
    unit = ErmentroutKopellPopulation(tau=2.0, delta=0.7, j=12.0, dt=0.005)
    for _ in range(100):
        unit.step(1.5)
    unit.reset()
    assert (unit.r, unit.v) == (0.1, -2.0)
    assert (unit.tau, unit.delta, unit.j, unit.dt) == (2.0, 0.7, 12.0, 0.005)


def test_python_batch_returns_both_states_and_consistent_finals() -> None:
    drive = np.linspace(0.5, 2.5, 64)
    unit = ErmentroutKopellPopulation(
        r=0.13,
        v=-1.7,
        tau=1.3,
        delta=0.8,
        eta_bar=-4.2,
        j=12.5,
        dt=0.004,
    )
    result = unit.simulate(drive, backend="python")
    for key in ("r", "v"):
        trace = cast(npt.NDArray[np.float64], result[key])
        assert trace.shape == (64,)
        assert np.isfinite(trace).all()
    assert unit.r == result["r_final"] == cast(npt.NDArray[np.float64], result["r"])[-1]
    assert unit.v == result["v_final"] == cast(npt.NDArray[np.float64], result["v"])[-1]


def test_empty_python_batch_preserves_initial_state() -> None:
    unit = ErmentroutKopellPopulation(r=0.13, v=-1.7)
    result = unit.simulate([], backend="python")
    assert cast(npt.NDArray[np.float64], result["r"]).shape == (0,)
    assert cast(npt.NDArray[np.float64], result["v"]).shape == (0,)
    assert (unit.r, unit.v) == (0.13, -1.7)
    assert (result["r_final"], result["v_final"]) == (0.13, -1.7)

# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for strict dimensional checking in EquationNeuron

from __future__ import annotations

import pint
import pytest

from sc_neurocore.neurons._units import DimensionalError
from sc_neurocore.neurons.equation_builder import from_equations

UNIT_REGISTRY = pint.UnitRegistry()
MEGAOHM = 1e6 * UNIT_REGISTRY.ohm


def test_strict_units_accept_dimensionally_coherent_equation() -> None:
    neuron = from_equations(
        "dv/dt = (-(v - E_L) + R * I) / tau_m",
        threshold="v > v_threshold",
        reset="v = v_reset",
        params={
            "E_L": -65.0 * UNIT_REGISTRY.millivolt,
            "R": 100.0 * MEGAOHM,
            "tau_m": 10.0 * UNIT_REGISTRY.millisecond,
        },
        init={"v": -65.0 * UNIT_REGISTRY.millivolt},
        constants={
            "v_threshold": -50.0 * UNIT_REGISTRY.millivolt,
            "v_reset": -65.0 * UNIT_REGISTRY.millivolt,
        },
        dt=0.1 * UNIT_REGISTRY.millisecond,
        units="strict",
        input_unit=1.0 * UNIT_REGISTRY.nanoampere,
    )

    spikes = sum(neuron.step(I=2.0 * UNIT_REGISTRY.nanoampere) for _ in range(200))
    state = neuron.get_state()

    assert spikes > 0
    assert str(state["v"].units) == "millivolt"


def test_strict_units_reject_incoherent_tau_dimension() -> None:
    with pytest.raises(DimensionalError):
        from_equations(
            "dv/dt = (-(v - E_L) + R * I) / tau_m",
            threshold="v > v_threshold",
            reset="v = v_reset",
            params={
                "E_L": -65.0 * UNIT_REGISTRY.millivolt,
                "R": 100.0 * MEGAOHM,
                "tau_m": 10.0 * UNIT_REGISTRY.picoampere / UNIT_REGISTRY.nanosiemens,
            },
            init={"v": -65.0 * UNIT_REGISTRY.millivolt},
            constants={
                "v_threshold": -50.0 * UNIT_REGISTRY.millivolt,
                "v_reset": -65.0 * UNIT_REGISTRY.millivolt,
            },
            dt=0.1 * UNIT_REGISTRY.millisecond,
            units="strict",
            input_unit=1.0 * UNIT_REGISTRY.nanoampere,
        )


def test_default_units_path_stays_numeric() -> None:
    neuron = from_equations(
        "dv/dt = -(v - v_rest) / tau_m + I",
        params={"v_rest": 0.0, "tau_m": 10.0},
        init={"v": 0.0},
        dt=0.1,
        units="none",
    )

    for _ in range(50):
        neuron.step(I=1.0)

    assert isinstance(neuron.get_state()["v"], float)


def test_strict_units_require_input_unit_when_input_is_referenced() -> None:
    with pytest.raises(ValueError, match="input_unit"):
        from_equations(
            "dv/dt = (-(v - E_L) + R * I) / tau_m",
            params={
                "E_L": -65.0 * UNIT_REGISTRY.millivolt,
                "R": 100.0 * MEGAOHM,
                "tau_m": 10.0 * UNIT_REGISTRY.millisecond,
            },
            init={"v": -65.0 * UNIT_REGISTRY.millivolt},
            dt=0.1 * UNIT_REGISTRY.millisecond,
            units="strict",
        )

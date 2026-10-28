# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — General power dimensions in strict equation models
"""Execute real strict-unit powers after conversion from display quantities."""

from __future__ import annotations

import pytest

from sc_neurocore.neurons._units import DimensionalError, UNIT_REGISTRY as u
from sc_neurocore.neurons.equation_builder import from_equations


@pytest.mark.parametrize("exponent", [-3.0, 1.5, 9.0])
def test_literal_powers_preserve_physical_dimensions(exponent: float) -> None:
    """The power applies to base-unit magnitudes and carries its exact dimension."""
    model = from_equations(
        f"dv/dt=gain*offset**{exponent!r}",
        init={"v": 0 * u.volt},
        params={"gain": 1 * u.volt ** (1 - exponent) / u.second, "offset": 2000 * u.millivolt},
        dt=0.125 * u.second,
        units="strict",
    )
    model.step()
    assert model.get_state()["v"].to(u.volt).magnitude == pytest.approx(0.125 * 2**exponent)


def test_dimensionless_dynamic_exponent_runs_in_strict_units() -> None:
    """A variable exponent is meaningful when both its value and base are dimensionless."""
    model = from_equations(
        "dv/dt=gain*base**alpha",
        init={"v": 0 * u.volt},
        params={
            "gain": 1 * u.volt / u.second,
            "base": 4 * u.dimensionless,
            "alpha": 0.5 * u.dimensionless,
        },
        dt=0.125 * u.second,
        units="strict",
    )
    model.step()
    assert model.get_state()["v"].to(u.volt).magnitude == 0.25


def test_dimensioned_exponent_refuses_at_model_admission() -> None:
    """Physical units cannot be used as an exponent of a real-valued power."""
    with pytest.raises(DimensionalError):
        from_equations(
            "dv/dt=gain*base**alpha",
            init={"v": 0 * u.volt},
            params={
                "gain": 1 * u.volt / u.second,
                "base": 4 * u.dimensionless,
                "alpha": 1 * u.second,
            },
            dt=0.125 * u.second,
            units="strict",
        )

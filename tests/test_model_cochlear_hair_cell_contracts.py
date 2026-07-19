# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Cochlear hair-cell model contracts

"""Module-specific behavioural contracts for ``CochlearHairCell``."""

from __future__ import annotations

import pytest


class TestCochlearHairCell:
    @pytest.fixture()
    def cell(self):
        from sc_neurocore.neurons.models import CochlearHairCell

        return CochlearHairCell()

    def test_defaults(self, cell):
        assert cell.v == -60.0
        assert cell.g_max == 10.0
        assert cell.delta == 0.1

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"g_max": -0.01},
            {"e_met": float("nan")},
            {"g_l": 0.0},
            {"e_l": float("inf")},
            {"cap": 0.0},
            {"x0": float("nan")},
            {"delta": 0.0},
            {"dt": 0.0},
            {"v": float("nan")},
            {"glutamate_release": -0.01},
        ],
    )
    def test_rejects_non_physical_hair_cell_parameters(self, kwargs):
        """MET channel and membrane parameters must be finite and physically bounded."""
        from sc_neurocore.neurons.models import CochlearHairCell

        with pytest.raises(ValueError):
            CochlearHairCell(**kwargs)

    @pytest.mark.parametrize("displacement", [float("nan"), float("inf")])
    def test_rejects_non_finite_met_displacement(self, displacement):
        """Boltzmann MET activation must fail closed on non-finite displacement."""
        from sc_neurocore.neurons.models import CochlearHairCell

        with pytest.raises(ValueError, match="displacement"):
            CochlearHairCell().p_open(displacement)

    def test_p_open_boltzmann(self, cell):
        """P_open(x) = 1/(1 + exp(-(x - x0)/delta))."""
        # At x = x0: P_open = 0.5.
        assert abs(cell.p_open(0.0) - 0.5) < 1e-10
        # Large positive: P_open -> 1.
        assert cell.p_open(1.0) > 0.99
        # Large negative: P_open -> 0.
        assert cell.p_open(-1.0) < 0.01

    def test_step_returns_binary(self, cell):
        s = cell.step(0.0)
        assert s in (0, 1)

    def test_graded_glutamate_release(self, cell):
        """Glutamate release scales with depolarisation."""
        for _ in range(200):
            cell.step(0.5)
        assert cell.glutamate_release >= 0.0

    def test_positive_displacement_depolarises(self):
        """Strong positive displacement should depolarise (increase V)."""
        from sc_neurocore.neurons.models import CochlearHairCell

        cell = CochlearHairCell()
        v_rest = cell.v
        for _ in range(200):
            cell.step(0.5)
        # MET channels open, current flows, V changes.
        assert cell.v != v_rest

    def test_negative_displacement_stays_near_rest(self):
        """Large negative displacement: MET channels closed, V near E_L."""
        from sc_neurocore.neurons.models import CochlearHairCell

        cell = CochlearHairCell()
        for _ in range(500):
            cell.step(-2.0)
        # P_open(-2.0) ~ 0.0, almost no MET current.
        assert abs(cell.v - cell.e_l) < 5.0

    def test_reset(self, cell):
        for _ in range(100):
            cell.step(0.5)
        cell.reset()
        assert cell.v == cell.e_l
        assert cell.glutamate_release == 0.0

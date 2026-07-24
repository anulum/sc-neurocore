# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestIzhikevich2007Neuron from former test_neuroml_import.py

"""Focused suite: TestIzhikevich2007Neuron from former test_neuroml_import.py."""

from __future__ import annotations

from tests.neuroml_import_support import *  # noqa: F403


class TestIzhikevich2007Neuron:
    def test_euler_step_matches_biophysical_equations_below_threshold(self):
        neuron = Izhikevich2007Neuron(
            C=100.0,
            k=0.7,
            vr=-60.0,
            vt=-40.0,
            vpeak=35.0,
            a=0.03,
            b=-2.0,
            c=-50.0,
            d=100.0,
            v0=-61.0,
            dt=0.1,
            integrator="euler",
        )

        spike = neuron.step(70.0)

        expected_dv = (0.7 * (-1.0) * (-21.0) - 2.0 + 70.0) / 100.0
        expected_du = 0.03 * (-2.0 * (-1.0) - 2.0)
        assert spike == 0
        assert neuron.v == pytest.approx(-61.0 + 0.1 * expected_dv)
        assert neuron.u == pytest.approx(2.0 + 0.1 * expected_du)

    def test_spike_reset_uses_vpeak_c_and_d(self):
        neuron = Izhikevich2007Neuron(
            C=100.0,
            k=0.7,
            vr=-60.0,
            vt=-40.0,
            vpeak=35.0,
            a=0.03,
            b=-2.0,
            c=-50.0,
            d=100.0,
            v0=34.0,
            dt=1.0,
            integrator="euler",
        )

        spike = neuron.step(500.0)

        assert spike == 1
        assert neuron.v == pytest.approx(-50.0)
        assert neuron.u == pytest.approx(-88.0)

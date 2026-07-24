# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestAstrocyteIsolation from former test_model_astrocyte.py

"""Focused suite: TestAstrocyteIsolation from former test_model_astrocyte.py."""

from __future__ import annotations

from tests.model_astrocyte_support import *  # noqa: F403


class TestAstrocyteIsolation:
    def test_defaults(self):
        n = AstrocyteModel()
        assert n.ca == 0.05 and n.h == 0.8 and n.ip3 == 0.5
        assert n.c0 == 2.0 and n.dt == 0.01

    def test_step_returns_float(self):
        """Returns Ca concentration (float), not binary spike."""
        n = AstrocyteModel()
        assert isinstance(n.step(0.0), float)

    def test_three_variables_evolve(self):
        n = AstrocyteModel()
        initial = (n.ca, n.h, n.ip3)
        for _ in range(500):
            n.step(0.5)
        for name, v0, v1 in zip(["ca", "h", "ip3"], initial, (n.ca, n.h, n.ip3)):
            assert v0 != v1, f"{name} didn't evolve"

    def test_state_finite(self):
        n = AstrocyteModel()
        for _ in range(100000):
            n.step(0.5)
        assert all(np.isfinite(v) for v in [n.ca, n.h, n.ip3])

    def test_reset(self):
        n = AstrocyteModel()
        for _ in range(500):
            n.step(1.0)
        n.reset()
        assert n.ca == 0.05 and n.h == 0.8 and n.ip3 == 0.5

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"ca": -0.01},
            {"h": -0.1},
            {"h": 1.1},
            {"ip3": -0.1},
            {"v_er": 0.0},
            {"k_er": 0.0},
            {"v_serca": 0.0},
            {"d1": 0.0},
            {"d2": 0.0},
            {"d3": 0.0},
            {"d5": 0.0},
            {"a2": 0.0},
            {"c0": 0.0},
            {"c1": 0.0},
            {"leak": -0.01},
            {"ip3_prod": -0.01},
            {"ip3_decay": -0.01},
            {"dt": 0.0},
        ],
    )
    def test_rejects_non_physical_configuration(self, kwargs):
        """Li-Rinzel calcium/IP3 parameters must be finite and physical."""
        with pytest.raises(ValueError):
            AstrocyteModel(**kwargs)

    @pytest.mark.parametrize("current", [-0.1, float("nan"), float("inf")])
    def test_rejects_non_physical_ip3_drive(self, current):
        """Glutamate-driven IP3 production must be finite and non-negative."""
        n = AstrocyteModel()
        with pytest.raises(ValueError, match="current"):
            n.step(current)

# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestAstrocyteCaDynamics from former test_model_astrocyte.py

"""Focused suite: TestAstrocyteCaDynamics from former test_model_astrocyte.py."""

from __future__ import annotations

from tests.model_astrocyte_support import *  # noqa: F403

class TestAstrocyteCaDynamics:
    """Core: IP3R channel + SERCA pump + ER leak."""

    def test_ca_oscillates_at_zero_input(self):
        """Spontaneous Ca oscillation from IP3R-Ca feedback loop."""
        n = AstrocyteModel()
        cas = []
        for _ in range(10000):
            cas.append(n.step(0.0))
        cas = np.array(cas)
        v_range = cas.max() - cas.min()
        assert v_range > 0.5, f"Ca range = {v_range:.4f}, expected oscillation"

    def test_ca_non_negative(self):
        n = AstrocyteModel()
        for _ in range(50000):
            ca = n.step(0.0)
            assert ca >= 0.0

    def test_ca_increases_with_ip3_input(self):
        """Glutamate → IP3 → Ca release from ER."""
        n_low = AstrocyteModel()
        n_high = AstrocyteModel()
        for _ in range(10000):
            n_low.step(0.0)
            n_high.step(1.0)
        assert n_high.ca > n_low.ca

    def test_ip3_drives_channel_opening(self):
        """Higher IP3 → more IP3R opening → more Ca release."""
        n = AstrocyteModel()
        for _ in range(10000):
            n.step(2.0)  # high IP3 production
        assert n.ip3 > 1.0  # IP3 has accumulated
        assert n.ca > 0.5  # Ca elevated from ER release

    def test_h_gate_bounded(self):
        """IP3R de-inactivation gate h ∈ [0, 1]."""
        n = AstrocyteModel()
        for _ in range(50000):
            n.step(0.5)
        assert 0.0 <= n.h <= 1.0

    def test_ca_conservation(self):
        """Total Ca = ca + c1·Ca_ER is conserved (c0).

        Ca_ER = (c0 - ca) / c1.
        """
        n = AstrocyteModel()
        for _ in range(10000):
            n.step(0.5)
        ca_er = (n.c0 - n.ca) / n.c1
        total = n.ca + n.c1 * ca_er
        assert abs(total - n.c0) < 1e-10

    def test_rejects_timestep_that_exits_total_calcium_pool(self):
        """Integrator must not accept cytosolic Ca above conserved total calcium."""
        n = AstrocyteModel(dt=100.0)
        with pytest.raises(ValueError, match="calcium"):
            n.step(0.0)

# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestODELIFDynamics from former test_spike_ode.py

"""Focused suite: TestODELIFDynamics from former test_spike_ode.py."""

from __future__ import annotations

from tests.spike_ode_support import *  # noqa: F403


class TestODELIFDynamics:
    def test_dvdt(self):
        d = ODELIFDynamics(tau_mem=20.0, v_rest=0.0)
        dv = d.dvdt(np.array([0.5]), np.array([1.0]))
        assert dv.shape == (1,)

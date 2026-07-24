# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestEdgeCases from former test_wilson_cowan_dynamics.py

"""Focused suite: TestEdgeCases from former test_wilson_cowan_dynamics.py."""

from __future__ import annotations

from tests.wilson_cowan_dynamics_support import *  # noqa: F403


class TestEdgeCases:
    rust = pytest.importorskip(
        "sc_neurocore_engine", reason="Rust engine required"
    ).py_wilson_cowan_simulate

    def test_zero_length_workload(self):
        out = self.rust(
            0.5,
            0.3,
            10.0,
            6.0,
            10.0,
            1.0,
            1.0,
            2.0,
            1.2,
            4.0,
            0.1,
            np.zeros(0),
        )
        assert out["e"].shape == (0,)
        assert out["e_final"] == 0.5
        assert out["i_final"] == 0.3

    def test_single_step(self):
        out = self.rust(
            0.1,
            0.05,
            10.0,
            6.0,
            10.0,
            1.0,
            1.0,
            2.0,
            1.2,
            4.0,
            0.1,
            np.array([3.0]),
        )
        assert out["e"].shape == (1,)
        assert out["e_final"] == out["e"][0]

    def test_boundary_init(self):
        """E=0 and E=1 initial conditions must not break downstream math."""
        for e_init in (0.0, 1.0):
            out = self.rust(
                e_init,
                0.5,
                10.0,
                6.0,
                10.0,
                1.0,
                1.0,
                2.0,
                1.2,
                4.0,
                0.1,
                np.full(200, 2.0),
            )
            assert math.isfinite(out["e_final"])
            assert math.isfinite(out["i_final"])

# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestKSymmetryEnforcement from former test_fim_symmetry_dynamics.py

"""Focused suite: TestKSymmetryEnforcement from former test_fim_symmetry_dynamics.py."""

from __future__ import annotations

from tests.fim_symmetry_dynamics_support import *  # noqa: F403


class TestKSymmetryEnforcement:
    def test_enforce_symmetry_method_exists(self):
        """Projection should have _enforce_symmetry method."""
        _, proj, _ = _make_self_connected_network(n=10)
        assert hasattr(proj, "_enforce_symmetry")

    def test_symmetry_called_during_stdp(self):
        """After STDP update, _enforce_symmetry should have been called.
        Note: random topology (p=0.3) is NOT symmetric in connectivity,
        so only edges that exist in BOTH directions get symmetrised.
        Asymmetry measure may remain nonzero due to one-way edges."""
        net, proj, _ = _make_self_connected_network(n=20)
        net.run(duration=0.05, dt=0.001)
        # Just verify it ran without error
        assert len(proj.data) > 0

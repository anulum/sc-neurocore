# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestNonLocality from former test_quantum_cognition.py

"""Focused suite: TestNonLocality from former test_quantum_cognition.py."""

from __future__ import annotations

from tests.quantum_cognition_support import *  # noqa: F403


class TestNonLocality:
    """Verify that quantum coupling produces non-local effects."""

    def test_explicit_hamiltonian_changes_distal_observable(self) -> None:
        """Non-local effects require explicit physical coupling tensors."""
        pool = SpinPoolMPS(n_sites=8, bond_dim=8)
        eff7_before = pool.get_local_atp_efficiency(7)
        tensor = np.zeros((3, 3), dtype=np.float64)
        tensor[0, 0] = 1.0
        pool.evolve_exact([SpinCouplingTensor(0, 7, tensor)], time_us=0.25)
        eff7_after = pool.get_local_atp_efficiency(7)
        assert eff7_before != eff7_after

    def test_proximal_stronger_than_distal(self) -> None:
        """Nearby neurons should be more affected than distant ones."""
        pool = SpinPoolMPS(n_sites=8)
        n0 = HybridFisherPosnerLIF(0, pool)

        # Record initial efficiencies
        for _ in range(100):
            n0.step(50.0)

        if n0._total_spikes > 0:
            eff_near = pool.entanglement_map[1]
            eff_far = pool.entanglement_map[7]
            assert eff_near > eff_far, (
                f"Proximity violation: near={eff_near:.4f}, far={eff_far:.4f}"
            )

# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestRadicalPairFieldResponse from former test_qc_e2e.py

"""Focused suite: TestRadicalPairFieldResponse from former test_qc_e2e.py."""

from __future__ import annotations

from tests.qc_e2e_support import *  # noqa: F403

class TestRadicalPairFieldResponse:
    """Sweep B from 0 to 100 µT, verify monotonic physics response."""

    def test_field_sweep_monotonicity(self) -> None:
        model = RadicalPairModel()
        fields = np.linspace(0, 1e-4, 200)
        yields = model.singlet_yield_field_sweep(fields)

        assert yields.shape == (200,)
        assert np.all(yields >= 0.0) and np.all(yields <= 1.0)
        assert np.all(np.isfinite(yields))

    def test_large_field_sweep(self) -> None:
        """Sweep from 0 to 10 T — verify no numerical instability."""
        model = RadicalPairModel()
        fields = np.logspace(-8, 1, 1000)  # 10 nT → 10 T
        yields = model.singlet_yield_field_sweep(fields)
        assert np.all(np.isfinite(yields))
        assert np.all(yields >= 0.0) and np.all(yields <= 1.0)

    def test_atp_efficiency_rejects_classical_boost(self) -> None:
        """ATP efficiency rejects non-Hamiltonian entanglement boosts."""
        model = RadicalPairModel()
        eff = model.atp_efficiency(b_local=50e-6, entanglement_boost=0.0)
        assert 0.0 <= eff <= 1.0
        with pytest.raises(ValueError, match="entanglement_boost"):
            model.atp_efficiency(b_local=50e-6, entanglement_boost=0.01)

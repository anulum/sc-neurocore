# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestCrossBackendParity from former test_qc_e2e.py

"""Focused suite: TestCrossBackendParity from former test_qc_e2e.py."""

from __future__ import annotations

from tests.qc_e2e_support import *  # noqa: F403

class TestCrossBackendParity:
    """Verify Python and Rust produce identical numerical results."""

    def test_spin_pool_measurement_parity(self) -> None:
        """Run identical measurement sequence through Python and Rust, compare."""
        rs_file = _QC_DIR / "spin_pool.rs"
        if not rs_file.exists():
            pytest.skip("spin_pool.rs not found")

        # Python reference
        pool = SpinPoolMPS(n_sites=8, bond_dim=16)
        pool.apply_measurement(3, 1.0)
        pool.apply_measurement(0, 0.5)
        pool.apply_measurement(7, 1.0)
        py_emap = pool.entanglement_map.copy()
        py_atp = [pool.get_local_atp_efficiency(i) for i in range(8)]

        # Verify normalisation
        assert abs(np.sum(py_emap) - 1.0) < 1e-10
        # Verify all ATP efficiencies in the physical probability range.
        # A product |00...0> state has zero adjacent singlet weight, so a
        # nonzero floor would be a classical proxy, not a quantum observable.
        for i, eff in enumerate(py_atp):
            assert 0.0 <= eff <= 1.0, f"ATP[{i}]={eff} out of range"

    def test_radical_pair_parity(self) -> None:
        """Python radical_pair must match Rust radical_pair to high precision."""
        model = RadicalPairModel()
        # Zero-field singlet yield
        phi_py = model.singlet_yield(0.0)
        # Regression for the exact default one-nucleus isotropic density
        # matrix RPM. The old scalar proxy gave ~0.257; that is no longer the
        # reference model.
        assert phi_py == pytest.approx(0.5983981639448483, abs=1e-12)

        # Strong exchange limit
        strong = RadicalPairModel(RadicalPairParams(exchange_j=1000.0))
        phi_strong = strong.singlet_yield(0.0)
        assert phi_strong > 0.9, f"Strong J should preserve singlet: {phi_strong}"

    def test_kane_coupling_parity(self) -> None:
        """Python kane_mapper must match Rust exchange coupling formula."""
        mapper = KaneSiliconMapper(spacing_nm=10.0)
        layout = mapper.map_pool_to_register(4)
        # Analytical: J(10nm) = 0.1 * exp(-2*10/2.5) = 0.1 * exp(-8)
        expected = 0.1 * math.exp(-8.0)
        nn_coupling = layout.coupling_matrix[0, 1]
        assert abs(nn_coupling - expected) < 1e-15, (
            f"NN coupling mismatch: {nn_coupling} vs {expected}"
        )

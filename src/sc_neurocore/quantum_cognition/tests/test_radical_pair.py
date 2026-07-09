# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Source/config provenance header


"""Inline tests for RadicalPairModel."""

from __future__ import annotations

import numpy as np
import pytest

from sc_neurocore.quantum_cognition.radical_pair import (
    RadicalPairModel,
    RadicalPairParams,
)


class TestRadicalPairParams:
    def test_defaults(self) -> None:
        p = RadicalPairParams()
        assert p.hyperfine_a == 10.0
        assert p.exchange_j == 1.0
        assert p.lifetime_us == 100.0

    def test_custom(self) -> None:
        p = RadicalPairParams(hyperfine_a=50.0, exchange_j=5.0)
        assert p.hyperfine_a == 50.0


class TestRadicalPairModel:
    def test_singlet_yield_zero_field(self) -> None:
        """At zero field, singlet yield depends only on exchange/hyperfine ratio."""
        model = RadicalPairModel()
        phi = model.singlet_yield(b_local=0.0)
        assert 0.0 <= phi <= 1.0

    def test_singlet_yield_range(self) -> None:
        """Singlet yield must be bounded [0, 1]."""
        model = RadicalPairModel()
        for b in [0.0, 1e-6, 50e-6, 1e-3, 1.0]:
            phi = model.singlet_yield(b)
            assert 0.0 <= phi <= 1.0, f"Out of range at B={b}: {phi}"

    def test_strong_exchange_preserves_singlet(self) -> None:
        """Strong exchange coupling should preserve singlet character."""
        params = RadicalPairParams(exchange_j=1000.0, hyperfine_a=1.0)
        model = RadicalPairModel(params)
        phi = model.singlet_yield(0.0)
        assert phi > 0.9, f"Strong J should give high singlet yield: {phi}"

    def test_weak_exchange_reduces_singlet(self) -> None:
        """Weak exchange coupling leads to more mixing → lower singlet yield."""
        strong = RadicalPairModel(RadicalPairParams(exchange_j=100.0))
        weak = RadicalPairModel(RadicalPairParams(exchange_j=0.01))
        assert strong.singlet_yield(0.0) > weak.singlet_yield(0.0)

    def test_field_sweep(self) -> None:
        """Field sweep should return array of correct length."""
        model = RadicalPairModel()
        fields = np.linspace(0, 1e-3, 50)
        yields = model.singlet_yield_field_sweep(fields)
        assert yields.shape == (50,)
        assert np.all(yields >= 0.0) and np.all(yields <= 1.0)

    def test_atp_efficiency_rejects_entanglement_boost(self) -> None:
        """Classical boost is not a radical-pair Hamiltonian parameter."""
        model = RadicalPairModel()
        with pytest.raises(ValueError, match="entanglement_boost"):
            model.atp_efficiency(b_local=0.0, entanglement_boost=0.3)

    def test_atp_efficiency_range(self) -> None:
        """ATP efficiency is singlet-yield-derived and bounded."""
        model = RadicalPairModel()
        eff = model.atp_efficiency(0.0)
        assert 0.0 <= eff <= 1.0

    def test_get_state(self) -> None:
        """State dict should contain all params."""
        model = RadicalPairModel()
        state = model.get_state()
        assert "hyperfine_a_mhz" in state
        assert "exchange_j_mhz" in state
        assert "lifetime_us" in state

    def test_repr(self) -> None:
        model = RadicalPairModel()
        r = repr(model)
        assert "RadicalPairModel" in r
        assert "MHz" in r

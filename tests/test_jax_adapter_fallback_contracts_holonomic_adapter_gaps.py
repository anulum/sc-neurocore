# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestHolonomicAdapterGaps from former test_jax_adapter_fallback_contracts.py

"""Focused suite: TestHolonomicAdapterGaps from former test_jax_adapter_fallback_contracts.py."""

from __future__ import annotations


class TestHolonomicAdapterGaps:
    """Remaining holonomic adapter paths."""

    def test_l2_step_no_inputs(self) -> None:
        """A step without inputs still gives an output."""
        from sc_neurocore.adapters.holonomic.l2_chem import L2_NeurochemicalAdapter

        a = L2_NeurochemicalAdapter()
        out = a.step_jax(0.01, inputs=None)
        assert out.shape[0] > 0

    def test_l4_decode(self) -> None:
        """The L4 decode reports synchronisation."""
        from sc_neurocore.adapters.holonomic.l4_cell import L4_CellularAdapter

        a = L4_CellularAdapter()
        bs = a.encode(None)
        d = a.decode(bs)
        assert "synchronization_r4" in d

    def test_l5_decode(self) -> None:
        """The L5 decode reports organismal valence."""
        from sc_neurocore.adapters.holonomic.l5_org import L5_OrganismalAdapter

        a = L5_OrganismalAdapter()
        bs = a.encode(None)
        d = a.decode(bs)
        assert "organismal_valence" in d

    def test_l6_step_no_inputs(self) -> None:
        """A step without inputs still gives an output."""
        from sc_neurocore.adapters.holonomic.l6_plan import L6_PlanetaryAdapter

        a = L6_PlanetaryAdapter()
        out = a.step_jax(0.01, inputs=None)
        assert out.shape[0] > 0

    def test_l6_decode(self) -> None:
        """The L6 decode reports the global coherence index."""
        from sc_neurocore.adapters.holonomic.l6_plan import L6_PlanetaryAdapter

        a = L6_PlanetaryAdapter()
        bs = a.encode(None)
        d = a.decode(bs)
        assert "global_coherence_index" in d

    def test_l7_step_no_inputs(self) -> None:
        """A step without inputs still gives an output."""
        from sc_neurocore.adapters.holonomic.l7_sym import L7_SymbolicAdapter

        a = L7_SymbolicAdapter()
        out = a.step_jax(0.01, inputs=None)
        assert out.shape[0] > 0

    def test_l7_decode(self) -> None:
        """The L7 decode reports symbolic unity."""
        from sc_neurocore.adapters.holonomic.l7_sym import L7_SymbolicAdapter

        a = L7_SymbolicAdapter()
        bs = a.encode(None)
        d = a.decode(bs)
        assert "symbolic_unity_r7" in d

    def test_l8_decode(self) -> None:
        """The L8 decode reports cosmic alignment."""
        from sc_neurocore.adapters.holonomic.l8_cosm import L8_CosmicAdapter

        a = L8_CosmicAdapter()
        bs = a.encode(None)
        d = a.decode(bs)
        assert "cosmic_alignment_r8" in d

    def test_l9_decode(self) -> None:
        """The L9 decode reports memory retrieval."""
        from sc_neurocore.adapters.holonomic.l9_mem import L9_MemoryAdapter

        a = L9_MemoryAdapter()
        bs = a.encode(None)
        d = a.decode(bs)
        assert "memory_retrieval_r9" in d

    def test_l10_step_no_inputs(self) -> None:
        """A step without inputs still gives an output."""
        from sc_neurocore.adapters.holonomic.l10_fire import L10_FirewallAdapter

        a = L10_FirewallAdapter()
        out = a.step_jax(0.01, inputs=None)
        assert out.shape[0] > 0

    def test_l10_decode(self) -> None:
        """The L10 decode reports firewall integrity."""
        from sc_neurocore.adapters.holonomic.l10_fire import L10_FirewallAdapter

        a = L10_FirewallAdapter()
        bs = a.encode(None)
        d = a.decode(bs)
        assert "firewall_integrity_r10" in d

    def test_l11_step_no_inputs(self) -> None:
        """A step without inputs still gives an output."""
        from sc_neurocore.adapters.holonomic.l11_noos import L11_NoosphericAdapter

        a = L11_NoosphericAdapter()
        out = a.step_jax(0.01, inputs=None)
        assert out.shape[0] > 0

    def test_l11_decode(self) -> None:
        """The L11 decode reports polarisation and coherence."""
        from sc_neurocore.adapters.holonomic.l11_noos import L11_NoosphericAdapter

        a = L11_NoosphericAdapter()
        bs = a.encode(None)
        d = a.decode(bs)
        assert "noospheric_polarization" in d
        assert "collective_coherence_r11" in d

    def test_l12_step_no_inputs(self) -> None:
        """A step without inputs still gives an output."""
        from sc_neurocore.adapters.holonomic.l12_gaian import L12_GaianAdapter

        a = L12_GaianAdapter()
        out = a.step_jax(0.01, inputs=None)
        assert out.shape[0] > 0

    def test_l12_decode(self) -> None:
        """The L12 decode reports Gaian synchrony."""
        from sc_neurocore.adapters.holonomic.l12_gaian import L12_GaianAdapter

        a = L12_GaianAdapter()
        bs = a.encode(None)
        d = a.decode(bs)
        assert "gaian_synchrony_index" in d

# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for SCPN layer helper contracts

"""Contracts for SCPN layer helper state accessors and bounded history behaviour."""

from __future__ import annotations

import numpy as np


def test_l2_neurochemical_history_release_and_state_contracts() -> None:
    from sc_neurocore.scpn.layers.l2_neurochemical import (
        L2_NeurochemicalLayer,
        L2_StochasticParameters,
    )

    layer = L2_NeurochemicalLayer(L2_StochasticParameters(n_receptors=4, bitstream_length=32))
    for _ in range(105):
        layer.step(0.01)
    layer.release_neurotransmitter(0, 0.5)
    state = layer.get_neuromodulation_state()

    assert len(layer.history) <= 100
    assert layer.nt_concentrations[0] > 0
    assert {"dopamine", "serotonin"} <= set(state)


def test_l3_l4_helper_outputs_have_expected_types() -> None:
    from sc_neurocore.scpn.layers.l3_genomic import L3_GenomicLayer, L3_StochasticParameters
    from sc_neurocore.scpn.layers.l4_cellular import L4_CellularLayer, L4_StochasticParameters

    l3 = L3_GenomicLayer(L3_StochasticParameters(n_genes=4, bitstream_length=32))
    l3.step(0.01)
    l4 = L4_CellularLayer(L4_StochasticParameters(grid_size=(4, 4), bitstream_length=32))
    l4.step(0.01)

    assert isinstance(l3.get_ciss_coherence(), float)
    assert l4.get_tissue_pattern().shape == (4, 4)


def test_l5_organismal_emotional_event_and_hrv_accessors() -> None:
    from sc_neurocore.scpn.layers.l5_organismal import (
        L5_OrganismalLayer,
        L5_StochasticParameters,
    )

    layer = L5_OrganismalLayer(
        L5_StochasticParameters(
            n_emotional_dims=8,
            n_autonomic_nodes=16,
            bitstream_length=32,
        )
    )
    output = layer.step(0.01, external_event={0: 0.5, 2: 0.3})
    layer.step(0.01)

    assert "output_bitstreams" in output
    assert isinstance(layer._compute_rmssd(), float)
    assert isinstance(layer.get_emotional_valence(), float)


def test_l6_l7_helper_outputs_are_bounded_and_structured() -> None:
    from sc_neurocore.scpn.layers.l6_ecological import L6_EcologicalLayer
    from sc_neurocore.scpn.layers.l6_ecological import L6_StochasticParameters
    from sc_neurocore.scpn.layers.l7_symbolic import L7_StochasticParameters
    from sc_neurocore.scpn.layers.l7_symbolic import L7_SymbolicLayer

    l6 = L6_EcologicalLayer(L6_StochasticParameters(n_field_nodes=4, bitstream_length=32))
    for _ in range(105):
        l6.step(0.01)
    spectrum = l6.get_schumann_spectrum()
    circadian_time = l6.get_circadian_time()

    l7 = L7_SymbolicLayer(L7_StochasticParameters(n_symbols=8, bitstream_length=32))
    l7.step(0.01)
    glyph = l7.get_glyph_vector_normalized()
    l7.stimulate_meridian(0, 0.5)
    acupoints = l7.get_acupoint_map()

    assert len(l6.history) <= 100
    assert isinstance(spectrum, dict)
    assert set(spectrum) == set(l6.params.schumann_frequencies)
    assert 0.0 <= circadian_time <= 24.0
    assert np.all(np.isfinite(glyph))
    assert np.all((glyph >= 0.0) & (glyph <= 1.0 + 1e-9))
    assert np.isclose(np.max(glyph), 1.0) or np.allclose(glyph, 0.0)
    assert isinstance(acupoints, dict)

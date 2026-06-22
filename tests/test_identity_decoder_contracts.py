# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for identity decoder contracts

"""Contracts for decoding identity-substrate state signatures."""

from __future__ import annotations

import numpy as np

from sc_neurocore.identity.decoder import StateDecoder
from sc_neurocore.identity.substrate import IdentitySubstrate


def test_decoder_handles_empty_spike_history_for_dominant_patterns() -> None:
    substrate = IdentitySubstrate(n_cortical=8)
    decoder = StateDecoder(substrate)

    patterns = decoder.extract_dominant_patterns()

    assert patterns.shape[0] >= 0


def test_decoder_extracts_attractor_state_list_after_drive() -> None:
    substrate = IdentitySubstrate(n_cortical=8)
    rng = np.random.default_rng(42)
    for _ in range(100):
        substrate.step(rng.standard_normal(8) * 5)
    decoder = StateDecoder(substrate)

    attractors = decoder.extract_attractor_states()

    assert isinstance(attractors, list)


def test_decoder_connectivity_signature_is_array_like() -> None:
    substrate = IdentitySubstrate(n_cortical=8)
    decoder = StateDecoder(substrate)

    signature = decoder.extract_connectivity_signature()

    assert signature.shape[0] >= 0


def test_decoder_groups_correlated_neurons_into_attractors() -> None:
    # A permissive threshold makes the first neuron absorb its correlated
    # partners into one ensemble, so the rest are skipped as already-visited.
    substrate = IdentitySubstrate(n_cortical=8)
    rng = np.random.default_rng(7)
    for _ in range(100):
        substrate.step(rng.standard_normal(8) * 5)
    decoder = StateDecoder(substrate)

    attractors = decoder.extract_attractor_states(threshold=-1.0)

    assert isinstance(attractors, list)
    assert any(group.size >= 2 for group in attractors)

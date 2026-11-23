# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestStateDecoder from former test_identity_lazarus.py

"""Focused suite: TestStateDecoder from former test_identity_lazarus.py."""

from __future__ import annotations

import numpy as np
from sc_neurocore.identity.decoder import (
    StateDecoder,
)
from sc_neurocore.identity.substrate import (
    IdentitySubstrate,
)


class TestStateDecoder:
    """Smoke tests of the state decoder."""

    def test_dominant_patterns_shape(self) -> None:
        """Dominant patterns are an array; their shape is not checked."""
        sub = IdentitySubstrate(n_cortical=50, n_inhibitory=20, n_memory=10, seed=42)
        sub.run(duration=0.05, dt=0.001)
        dec = StateDecoder(sub)
        patterns = dec.extract_dominant_patterns(n_components=3)
        assert isinstance(patterns, np.ndarray)

    def test_attractor_states(self) -> None:
        """Attractor states are a list; their contents are not checked."""
        sub = IdentitySubstrate(n_cortical=50, n_inhibitory=20, n_memory=10, seed=42)
        sub.run(duration=0.1, dt=0.001)
        dec = StateDecoder(sub)
        attractors = dec.extract_attractor_states(threshold=0.3)
        assert isinstance(attractors, list)

    def test_connectivity_signature(self) -> None:
        """The connectivity signature is an array."""
        sub = IdentitySubstrate(n_cortical=30, n_inhibitory=10, n_memory=5, seed=42)
        sub.run(duration=0.05, dt=0.001)
        dec = StateDecoder(sub)
        conn = dec.extract_connectivity_signature()
        assert isinstance(conn, np.ndarray)

    def test_priming_context_is_string(self) -> None:
        """The priming context is a string."""
        sub = IdentitySubstrate(n_cortical=30, n_inhibitory=10, n_memory=5, seed=42)
        sub.run(duration=0.05, dt=0.001)
        dec = StateDecoder(sub)
        ctx = dec.generate_priming_context()
        assert isinstance(ctx, str)

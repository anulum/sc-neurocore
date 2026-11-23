# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestDecoder from former test_identity_substrate.py

"""Focused suite: TestDecoder from former test_identity_substrate.py."""

from __future__ import annotations

import numpy as np
from sc_neurocore.identity import (
    StateDecoder,
)
from tests.identity_substrate_support import (
    _make_substrate,
    N_CORTICAL,
)


class TestDecoder:
    """Decoding the substrate's state."""

    def test_priming_context_dormant(self) -> None:
        """A fresh substrate reads as dormant."""
        sub = _make_substrate()
        dec = StateDecoder(sub)
        ctx = dec.generate_priming_context()
        assert "dormant" in ctx.lower() or "0 steps" in ctx

    def test_priming_context_after_activity(self) -> None:
        """An active substrate reads as active."""
        sub = _make_substrate()
        stim = np.random.default_rng(0).uniform(5, 15, (200, N_CORTICAL))
        sub.run(duration=0.2, dt=0.001, stimuli_sequence=stim)
        dec = StateDecoder(sub)
        ctx = dec.generate_priming_context()
        assert "active" in ctx.lower() or "steps" in ctx.lower()
        assert len(ctx) > 20

    def test_connectivity_signature_shape(self) -> None:
        """The connectivity signature is a square matrix."""
        sub = _make_substrate()
        stim = np.random.default_rng(0).uniform(5, 15, (200, N_CORTICAL))
        sub.run(duration=0.2, dt=0.001, stimuli_sequence=stim)
        dec = StateDecoder(sub)
        fc = dec.extract_connectivity_signature()
        assert fc.ndim == 2
        assert fc.shape[0] == fc.shape[1]

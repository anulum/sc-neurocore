# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestDeltaEncode from former test_encoding.py

"""Focused suite: TestDeltaEncode from former test_encoding.py."""

from __future__ import annotations

from tests.test_training.encoding_support import (
    delta_encode,
    torch,
)


class TestDeltaEncode:
    """Delta encoding of signal traces into spikes."""

    def test_output_shape_matches_input_trace(self) -> None:
        """The spikes keep the trace's shape."""
        values = torch.randn(10, 4)

        spikes = delta_encode(values, threshold=0.1)

        assert spikes.shape == (10, 4)

    def test_constant_signal_is_silent(self) -> None:
        """A constant signal emits no spikes."""
        values = torch.ones(10, 4) * 5.0

        spikes = delta_encode(values, threshold=0.1)

        assert spikes.sum().item() == 0.0

    def test_step_change_emits_at_transition(self) -> None:
        """A step change emits a spike at the transition."""
        values = torch.zeros(10, 1)
        values[5:] = 1.0

        spikes = delta_encode(values, threshold=0.5)

        assert spikes[5, 0].item() == 1.0

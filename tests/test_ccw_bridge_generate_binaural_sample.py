# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestGenerateBinauralSample from former test_ccw_bridge.py

"""Focused suite: TestGenerateBinauralSample from former test_ccw_bridge.py."""

from __future__ import annotations

import numpy as np
import pytest
from sc_neurocore.interfaces.ccw_bridge import (
    create_bridge,
)


class TestGenerateBinauralSample:
    """Generating binaural audio samples."""

    def test_shapes_match_requested_duration(self) -> None:
        """Both channels have the asked length."""
        bridge = create_bridge()
        left, right = bridge.generate_binaural_sample({"carrier_frequency": 432.0}, 256)
        assert left.shape == (256,)
        assert right.shape == (256,)

    def test_default_duration_is_1024(self) -> None:
        """The default length is 1024 samples."""
        bridge = create_bridge()
        left, right = bridge.generate_binaural_sample({})
        assert left.shape == (1024,)
        assert right.shape == (1024,)

    def test_amplitude_bounds_the_signal(self) -> None:
        """The amplitude bounds both channels."""
        bridge = create_bridge()
        amplitude = 0.3
        left, right = bridge.generate_binaural_sample({"amplitude": amplitude}, 512)
        assert np.max(np.abs(left)) <= amplitude + 1e-9
        assert np.max(np.abs(right)) <= amplitude + 1e-9

    def test_phase_state_is_continuous(self) -> None:
        """The phase carries over between calls."""
        bridge = create_bridge()
        bridge.generate_binaural_sample({"carrier_frequency": 100.0}, 128)
        first_phase = bridge.phase_left
        bridge.generate_binaural_sample({"carrier_frequency": 100.0}, 128)
        # Phase advances and is wrapped into [0, 2π).
        assert 0.0 <= bridge.phase_left < 2 * np.pi
        assert bridge.phase_left != pytest.approx(first_phase)

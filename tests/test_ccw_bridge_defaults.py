# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestDefaults from former test_ccw_bridge.py

"""Focused suite: TestDefaults from former test_ccw_bridge.py."""

from __future__ import annotations

import numpy as np
import pytest
from sc_neurocore.interfaces.ccw_bridge import (
    CCWBridge,
    CCWMode,
    CCWParameters,
    VIBRANAState,
)


class TestDefaults:
    """Default CCW parameters and state."""

    def test_ccw_parameters_defaults(self) -> None:
        """The default frequencies are the Schumann and carrier values."""
        p = CCWParameters()
        assert p.base_frequency == pytest.approx(7.83)  # Schumann
        assert p.carrier_frequency == pytest.approx(432.0)  # Verdi A4
        assert p.binaural_offset == pytest.approx(10.0)
        assert p.modulation_depth == pytest.approx(0.5)
        assert p.sample_rate == 44100

    def test_vibrana_state_defaults(self) -> None:
        """The visual state starts in meditation mode."""
        s = VIBRANAState()
        assert s.mode is CCWMode.MEDITATION
        assert s.geometry_phase == 0.0
        # glyph_weights default_factory produces a zeroed 6-vector.
        assert s.glyph_weights.shape == (6,)
        assert np.all(s.glyph_weights == 0.0)

    def test_ccw_mode_string_values(self) -> None:
        """Each mode has its string value."""
        assert CCWMode.THEURGIC.value == "theurgic"
        assert {m.value for m in CCWMode} == {
            "theurgic",
            "healing",
            "meditation",
            "cosmic",
            "focus",
            "creativity",
        }

    def test_init_uses_default_parameters_when_none(self) -> None:
        """Without parameters the bridge uses the defaults."""
        bridge = CCWBridge()
        assert isinstance(bridge.params, CCWParameters)
        assert bridge.params.sample_rate == 44100
        assert bridge.vibrana_state.mode is CCWMode.MEDITATION
        assert bridge.smoothing_window == 10

    def test_init_accepts_explicit_parameters(self) -> None:
        """Explicit parameters are kept."""
        params = CCWParameters(carrier_frequency=528.0, sample_rate=22050)
        bridge = CCWBridge(params)
        assert bridge.params is params
        assert bridge.params.carrier_frequency == pytest.approx(528.0)

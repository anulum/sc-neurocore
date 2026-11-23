# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestFactory from former test_ccw_bridge.py

"""Focused suite: TestFactory from former test_ccw_bridge.py."""

from __future__ import annotations

import pytest
from sc_neurocore.interfaces.ccw_bridge import (
    CCWBridge,
    CCWParameters,
    create_bridge,
)


class TestFactory:
    """The CCW bridge factory."""

    def test_create_bridge_without_params(self) -> None:
        """The factory builds a bridge with defaults."""
        bridge = create_bridge()
        assert isinstance(bridge, CCWBridge)
        assert isinstance(bridge.params, CCWParameters)

    def test_create_bridge_with_params(self) -> None:
        """The factory keeps given parameters."""
        params = CCWParameters(binaural_offset=6.0)
        bridge = create_bridge(params)
        assert bridge.params.binaural_offset == pytest.approx(6.0)

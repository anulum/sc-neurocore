# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestDVSLoader from former test_sensors_dvs.py

"""Focused suite: TestDVSLoader from former test_sensors_dvs.py."""

from __future__ import annotations

import numpy as np
import pytest
from sc_neurocore.sensors import (
    DVSLoader,
)
from tests.sensors_dvs_support import (
    _make_events,
)


class TestDVSLoader:
    """Loading event-camera recordings."""

    def test_n_pixels(self) -> None:
        """The pixel count is width times height."""
        loader = DVSLoader(width=8, height=6)
        assert loader.n_pixels == 48

    def test_default_dims(self) -> None:
        """The default sensor is 346 by 260."""
        loader = DVSLoader()
        assert loader.width == 346
        assert loader.height == 260

    def test_from_numpy_structured(self) -> None:
        """A structured event array passes through unchanged."""
        loader = DVSLoader(width=8, height=6)
        events = _make_events()
        result = loader.from_numpy(events)
        assert result is events  # passthrough for structured

    def test_from_numpy_2d_array(self) -> None:
        """A plain 2-D array becomes a structured event array."""
        loader = DVSLoader(width=8, height=6)
        raw = np.array(
            [
                [3, 2, 1000, 1],
                [5, 4, 2000, 0],
                [1, 0, 3000, 1],
            ],
            dtype=np.float64,
        )
        result = loader.from_numpy(raw)
        assert result.dtype.names is not None
        assert result["x"][0] == 3
        assert result["y"][1] == 4
        assert result["p"][2] == 1

    def test_from_numpy_invalid(self) -> None:
        """An array that cannot be structured is rejected."""
        loader = DVSLoader()
        with pytest.raises(ValueError, match="must be structured"):
            loader.from_numpy(np.array([1, 2, 3]))

    def test_from_tonic_import_error(self) -> None:
        """Without tonic the loader asks for it to be installed."""
        loader = DVSLoader()
        with pytest.raises(ImportError, match="pip install tonic"):
            loader.from_tonic("nmnist")

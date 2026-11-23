# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestWorldModelModuleExports from former test_world_model.py

"""Focused suite: TestWorldModelModuleExports from former test_world_model.py."""

from __future__ import annotations

import pytest
import sc_neurocore.world_model as world_model_module


class TestWorldModelModuleExports:
    """The lazy exports of the world-model package."""

    def test_lazy_exports_are_available(self) -> None:
        """The lazy exports resolve."""
        assert world_model_module.SCPlanner is not None
        assert world_model_module.PredictiveWorldModel is not None
        assert "SpikePredictor" in world_model_module.__all__

    def test_unknown_lazy_export_raises_attribute_error(self) -> None:
        """An unknown export raises AttributeError."""
        with pytest.raises(AttributeError):
            world_model_module.__getattr__("not_an_export")

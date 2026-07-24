# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestCreateFullStack from former test_scpn_integrated.py

"""Focused suite: TestCreateFullStack from former test_scpn_integrated.py."""

from __future__ import annotations

from tests.scpn_integrated_support import *  # noqa: F403


class TestCreateFullStack:
    def test_returns_dict(self):
        stack = create_full_stack()
        assert isinstance(stack, dict)

    def test_has_16_layers(self):
        stack = create_full_stack()
        assert len(stack) == 16

    def test_keys_match_registry(self):
        stack = create_full_stack()
        for key in LAYER_REGISTRY:
            assert key in stack

    def test_layers_have_step(self):
        """Each layer should have a step() method."""
        stack = create_full_stack()
        for key, layer in stack.items():
            assert hasattr(layer, "step"), f"{key} has no step() method"

    def test_layers_have_global_metric(self):
        stack = create_full_stack()
        for key, layer in stack.items():
            assert hasattr(layer, "get_global_metric"), f"{key} has no get_global_metric()"

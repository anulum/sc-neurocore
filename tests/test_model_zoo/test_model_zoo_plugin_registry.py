# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPluginRegistry from former test_model_zoo.py

"""Focused suite: TestPluginRegistry from former test_model_zoo.py."""

from __future__ import annotations

from model_zoo_support import *  # noqa: F403

class TestPluginRegistry:
    def test_register_and_get(self):
        reg = PluginRegistry()
        reg.register(LIFPlugin())
        assert "LIF" in reg
        assert reg.get("LIF") is not None

    def test_list_plugins_sorted(self):
        reg = PluginRegistry.with_builtins()
        names = reg.list_plugins()
        assert names == sorted(names)
        assert len(names) == 4

    def test_builtins_all_present(self):
        reg = PluginRegistry.with_builtins()
        assert "LIF" in reg
        assert "Izhikevich" in reg
        assert "AdEx" in reg
        assert "Hodgkin-Huxley" in reg

    def test_get_missing_returns_none(self):
        reg = PluginRegistry()
        assert reg.get("nonexistent") is None

    def test_len(self):
        reg = PluginRegistry()
        assert len(reg) == 0
        reg.register(LIFPlugin())
        assert len(reg) == 1

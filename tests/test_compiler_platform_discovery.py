# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Compiler platform-discovery contracts

"""Contracts for compiler platform registration and discovery hooks."""

from __future__ import annotations


class TestDiscoveryHook:
    def test_register_and_discover(self) -> None:
        from sc_neurocore.compiler.intelligence import (
            register_platform_hook,
            discover_platforms,
            _DISCOVERY_HOOKS,
        )
        from sc_neurocore.compiler.platforms import HardwareProfile

        def my_hook() -> list[HardwareProfile]:
            return [
                HardwareProfile(
                    name="test_discovered_chip",
                    vendor="HookVendor",
                    family="HookFam",
                    platform_class="custom",
                    data_width=16,
                    fraction=8,
                    overflow="saturate",
                    rounding="nearest",
                )
            ]

        register_platform_hook(my_hook)
        found = discover_platforms()
        assert "test_discovered_chip" in found
        # Cleanup
        _DISCOVERY_HOOKS.pop()

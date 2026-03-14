# SPDX-License-Identifier: AGPL-3.0-or-later
from __future__ import annotations

import pytest


class TestRegistryWiring:
    def test_all_16_adapters_registered(self):
        from sc_neurocore.utils.registry import registry
        import sc_neurocore.adapters.holonomic  # triggers registration

        adapters = registry.list("adapter")
        assert len(adapters) >= 16

    def test_create_adapter_factory(self):
        from sc_neurocore.adapters.holonomic import create_adapter

        for i in range(1, 17):
            adapter = create_adapter(i)
            assert adapter is not None

    def test_create_adapter_invalid_layer(self):
        from sc_neurocore.adapters.holonomic import create_adapter

        with pytest.raises(ValueError):
            create_adapter(0)
        with pytest.raises(ValueError):
            create_adapter(17)

    def test_registry_list(self):
        from sc_neurocore.utils.registry import registry
        import sc_neurocore.adapters.holonomic

        names = registry.list("adapter")
        assert "L1_Quantum" in names
        assert "L16_Meta" in names


class TestAdapterDiscovery:
    def test_discover_returns_dict(self):
        from sc_neurocore.utils.adapter_discovery import discover_adapters

        result = discover_adapters()
        assert isinstance(result, dict)

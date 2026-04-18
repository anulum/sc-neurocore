# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for Adapter Ecosystem

from __future__ import annotations

import pytest


class TestRegistryWiring:
    @pytest.fixture(autouse=True)
    def _ensure_registered(self):
        pass  # triggers registration

    def test_all_16_adapters_registered(self):
        from sc_neurocore.utils.registry import registry

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

        names = registry.list("adapter")
        assert "L1_Quantum" in names
        assert "L16_Meta" in names


class TestAdapterDiscovery:
    def test_discover_returns_dict(self):
        from sc_neurocore.utils.adapter_discovery import discover_adapters

        result = discover_adapters()
        assert isinstance(result, dict)

    def test_discover_with_mock_entry_point(self):
        from unittest.mock import MagicMock, patch

        from sc_neurocore.utils.adapter_discovery import discover_adapters

        mock_ep = MagicMock()
        mock_ep.name = "MockAdapterTest"
        mock_ep.load.return_value = type("MockAdapterTest", (), {})

        with patch(
            "sc_neurocore.utils.adapter_discovery.importlib.metadata.entry_points",
            return_value=[mock_ep],
        ):
            result = discover_adapters()
        assert isinstance(result, dict)

    def test_discover_handles_load_error(self):
        from unittest.mock import MagicMock, patch

        from sc_neurocore.utils.adapter_discovery import discover_adapters

        mock_ep = MagicMock()
        mock_ep.name = "BadAdapter"
        mock_ep.load.side_effect = ImportError("no such module")

        with patch(
            "sc_neurocore.utils.adapter_discovery.importlib.metadata.entry_points",
            return_value=[mock_ep],
        ):
            result = discover_adapters()
        assert "BadAdapter" not in result

    def test_discover_handles_old_api(self):
        from unittest.mock import MagicMock, patch

        from sc_neurocore.utils.adapter_discovery import discover_adapters

        call_count = 0

        def old_api(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if "group" in kwargs:
                raise TypeError("unexpected keyword argument 'group'")
            mock_result = MagicMock()
            mock_result.get.return_value = []
            return mock_result

        with patch(
            "sc_neurocore.utils.adapter_discovery.importlib.metadata.entry_points",
            side_effect=old_api,
        ):
            result = discover_adapters()
        assert isinstance(result, dict)
        assert call_count == 2  # first call raises, second succeeds

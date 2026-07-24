# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestRegistryWiring from former test_adapter_ecosystem.py

"""Focused suite: TestRegistryWiring from former test_adapter_ecosystem.py."""

from __future__ import annotations

from tests.adapter_ecosystem_support import *  # noqa: F403


class TestRegistryWiring:
    @pytest.fixture(autouse=True)
    def _ensure_registered(self) -> None:
        pass  # triggers registration

    def test_all_16_adapters_registered(self) -> None:
        from sc_neurocore.utils.registry import registry

        adapters = registry.list("adapter")
        assert len(adapters) >= 16

    def test_create_adapter_factory(self) -> None:
        from sc_neurocore.adapters.holonomic import create_adapter

        for i in range(1, 17):
            adapter = create_adapter(i)
            assert adapter is not None

    def test_create_adapter_invalid_layer(self) -> None:
        from sc_neurocore.adapters.holonomic import create_adapter

        with pytest.raises(ValueError):
            create_adapter(0)
        with pytest.raises(ValueError):
            create_adapter(17)

    def test_registry_list(self) -> None:
        from sc_neurocore.utils.registry import registry

        names = registry.list("adapter")
        assert "L1_Quantum" in names
        assert "L16_Meta" in names

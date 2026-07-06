# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for Adapter Ecosystem

from __future__ import annotations

import importlib
from pathlib import Path
import sys
from textwrap import dedent
from typing import cast

import numpy as np
import pytest

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


REPO_ROOT = Path(__file__).resolve().parents[1]


def _pyproject_adapter_entry_points(group: str) -> dict[str, str]:
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    entry_points = pyproject["project"]["entry-points"][group]
    assert isinstance(entry_points, dict)
    result: dict[str, str] = {}
    for name, target in entry_points.items():
        assert isinstance(name, str)
        assert isinstance(target, str)
        result[name] = target
    return result


def _resolve_entry_point_target(target: str) -> type:
    module_name, separator, attribute_path = target.partition(":")
    assert module_name
    assert separator == ":"
    assert attribute_path
    resolved: object = importlib.import_module(module_name)
    for attribute in attribute_path.split("."):
        resolved = getattr(resolved, attribute)
    assert isinstance(resolved, type)
    return resolved


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


class TestAdapterDiscovery:
    def test_first_party_entry_points_declared_in_pyproject(self) -> None:
        from sc_neurocore.utils.adapter_discovery import (
            ADAPTER_ENTRY_POINT_GROUP,
            FIRST_PARTY_ADAPTERS,
        )

        declared = _pyproject_adapter_entry_points(ADAPTER_ENTRY_POINT_GROUP)

        assert declared == FIRST_PARTY_ADAPTERS
        for target in declared.values():
            assert _resolve_entry_point_target(target).__module__.startswith(
                "sc_neurocore.adapters"
            )

    def test_discover_registers_first_party_importers(self) -> None:
        from sc_neurocore.utils.adapter_discovery import FIRST_PARTY_ADAPTERS, discover_adapters
        from sc_neurocore.utils.registry import registry

        registry.clear("adapter")
        try:
            discovered = discover_adapters(include_entry_points=False)
            registered = set(registry.list("adapter"))

            assert discovered.keys() >= FIRST_PARTY_ADAPTERS.keys()
            assert registered >= FIRST_PARTY_ADAPTERS.keys()
            assert registry.get("adapter", "neuroml").__name__ == "NeuroMLImporter"
            assert registry.get("adapter", "sonata").__name__ == "SONATAImporter"
            assert registry.get("adapter", "spikeinterface").__name__ == "SpikeInterfaceImporter"
        finally:
            registry.clear("adapter")
            sys.modules.pop("sc_neurocore.adapters.holonomic", None)
            importlib.import_module("sc_neurocore.adapters.holonomic")

    def test_discovery_keeps_holonomic_lazy_registry_available(self) -> None:
        from sc_neurocore.utils.adapter_discovery import discover_adapters
        from sc_neurocore.utils.registry import registry

        registry.clear("adapter")
        sys.modules.pop("sc_neurocore.adapters.holonomic", None)
        try:
            discover_adapters(include_entry_points=False)
            registered = registry.list("adapter")

            assert "neuroml" in registered
            assert "L1_Quantum" in registered
            assert "L16_Meta" in registered
        finally:
            registry.clear("adapter")
            sys.modules.pop("sc_neurocore.adapters.holonomic", None)
            importlib.import_module("sc_neurocore.adapters.holonomic")

    def test_discover_skips_invalid_first_party_targets(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from sc_neurocore.utils import adapter_discovery

        monkeypatch.setitem(adapter_discovery.FIRST_PARTY_ADAPTERS, "bad_format", "missing_colon")
        monkeypatch.setitem(
            adapter_discovery.FIRST_PARTY_ADAPTERS,
            "not_class",
            "sc_neurocore.adapters.spikeinterface:spike_trains_to_bitstreams",
        )

        discovered = adapter_discovery.discover_adapters(include_entry_points=False)

        assert "bad_format" not in discovered
        assert "not_class" not in discovered

    def test_first_party_importer_classes_delegate_to_real_adapters(self, tmp_path: Path) -> None:
        h5py = pytest.importorskip("h5py")
        from sc_neurocore.adapters.importers import (
            NeuroMLImporter,
            SONATAImporter,
            SpikeInterfaceImporter,
        )
        from sc_neurocore.utils.adapter_discovery import discover_adapters

        discovered = discover_adapters(include_entry_points=False)
        neuroml_cls = cast(type[NeuroMLImporter], discovered["neuroml"])
        sonata_cls = cast(type[SONATAImporter], discovered["sonata"])
        spikeinterface_cls = cast(type[SpikeInterfaceImporter], discovered["spikeinterface"])

        neuroml_path = tmp_path / "lif.nml"
        neuroml_path.write_text(
            dedent(
                """\
                <neuroml xmlns="http://www.neuroml.org/schema/neuroml2" id="test">
                  <iafTauCell id="lif" tau="20ms" leakReversal="-65mV"
                              thresh="-55mV" reset="-70mV"/>
                </neuroml>
                """
            ),
            encoding="utf-8",
        )
        cells = neuroml_cls.import_cells(neuroml_path)
        assert cells[0].cell_type == "StochasticLIFNeuron"
        assert neuroml_cls.create_neuron(cells[0]) is not None

        nodes_path = tmp_path / "nodes.h5"
        with h5py.File(nodes_path, "w") as h5_file:
            group = h5_file.create_group("nodes/exc")
            group.create_dataset("node_id", data=np.arange(2))
            group.create_dataset("node_type_id", data=np.zeros(2, dtype=int))
        network = sonata_cls.import_network(nodes_path)
        assert network.n_nodes == 2
        assert network.n_edges == 0

        spike_times = {0: np.array([0.0, 2.0]), 1: np.array([1.0])}
        bitstreams = spikeinterface_cls.to_bitstreams(spike_times, duration_ms=4.0)
        population_input = spikeinterface_cls.to_population_input(spike_times, duration_ms=4.0)
        probabilities = spikeinterface_cls.to_probabilities(spike_times, duration_ms=1000.0)
        assert bitstreams.shape == (2, 4)
        assert population_input.shape == (4, 2)
        np.testing.assert_allclose(probabilities, np.array([0.02, 0.01]))

    def test_adapter_package_exports_discovery_api(self) -> None:
        import sc_neurocore.adapters as adapters
        from sc_neurocore.utils.adapter_discovery import discover_adapters

        assert adapters.discover_adapters is discover_adapters

    def test_discover_returns_dict(self) -> None:
        from sc_neurocore.utils.adapter_discovery import discover_adapters

        result = discover_adapters()
        assert isinstance(result, dict)

    def test_discover_with_mock_entry_point(self) -> None:
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

    def test_discover_handles_load_error(self) -> None:
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

    def test_discover_handles_non_class_entry_point(self) -> None:
        from unittest.mock import MagicMock, patch

        from sc_neurocore.utils.adapter_discovery import discover_adapters

        mock_ep = MagicMock()
        mock_ep.name = "NotAClass"
        mock_ep.load.return_value = object()

        with patch(
            "sc_neurocore.utils.adapter_discovery.importlib.metadata.entry_points",
            return_value=[mock_ep],
        ):
            result = discover_adapters(include_first_party=False)
        assert result == {}

    def test_discover_handles_old_selectable_api(self) -> None:
        from unittest.mock import patch

        from sc_neurocore.utils.adapter_discovery import discover_adapters

        class SelectableEntryPoints:
            def select(self, *, group: str) -> list[object]:
                assert group == "sc_neurocore.adapters"
                return []

        call_count = 0

        def old_selectable_api(*args: object, **kwargs: object) -> object:
            nonlocal call_count
            call_count += 1
            if "group" in kwargs:
                raise TypeError("unexpected keyword argument 'group'")
            return SelectableEntryPoints()

        with patch(
            "sc_neurocore.utils.adapter_discovery.importlib.metadata.entry_points",
            side_effect=old_selectable_api,
        ):
            result = discover_adapters()
        assert isinstance(result, dict)
        assert call_count == 2

    def test_discover_handles_old_api(self) -> None:
        from unittest.mock import patch

        from sc_neurocore.utils.adapter_discovery import discover_adapters

        call_count = 0

        class LegacyEntryPoints:
            def get(self, group: str, default: object) -> list[object]:
                assert group == "sc_neurocore.adapters"
                assert default == ()
                return []

        def old_api(*args: object, **kwargs: object) -> object:
            nonlocal call_count
            call_count += 1
            if "group" in kwargs:
                raise TypeError("unexpected keyword argument 'group'")
            return LegacyEntryPoints()

        with patch(
            "sc_neurocore.utils.adapter_discovery.importlib.metadata.entry_points",
            side_effect=old_api,
        ):
            result = discover_adapters()
        assert isinstance(result, dict)
        assert call_count == 2  # first call raises, second succeeds

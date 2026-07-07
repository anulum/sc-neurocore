# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for the component registry

"""Tests for the component registry."""

from __future__ import annotations

import importlib
import sys

import pytest

from sc_neurocore.utils.registry import ComponentRegistry


@pytest.fixture
def reg():
    return ComponentRegistry()


def test_register_and_get(reg):
    @reg.register("neuron", "MyLIF")
    class MyLIF:
        pass

    assert reg.get("neuron", "MyLIF") is MyLIF


def test_register_uses_classname_if_no_name(reg):
    @reg.register("synapse")
    class CustomSynapse:
        pass

    assert reg.get("synapse", "CustomSynapse") is CustomSynapse


def test_duplicate_raises(reg):
    @reg.register("neuron", "LIF")
    class A:
        pass

    with pytest.raises(KeyError, match="already registered"):

        @reg.register("neuron", "LIF")
        class B:
            pass


def test_get_missing_raises(reg):
    with pytest.raises(KeyError, match="not registered"):
        reg.get("neuron", "NoSuch")


def test_list_namespace(reg):
    @reg.register("layer", "Dense")
    class D:
        pass

    @reg.register("layer", "Conv")
    class C:
        pass

    assert reg.list("layer") == ["Conv", "Dense"]


def test_list_empty_namespace(reg):
    assert reg.list("empty") == []


def test_singleton_list_lazy_loads_holonomic_adapters():
    from sc_neurocore.utils.registry import registry

    module_name = "sc_neurocore.adapters.holonomic"
    sys.modules.pop(module_name, None)

    registry.clear("adapter")
    try:
        adapters = registry.list("adapter")

        # The 16 holonomic L-layer adapters lazy-load on first listing. The registry now also
        # carries the entry-point-discovered adapters (importers + holonomic sub-adapters), so
        # assert the holonomic set is present rather than an exact total that grows whenever a
        # new `sc_neurocore.adapters` entry point is registered.
        holonomic = [a for a in adapters if a[:1] == "L" and a[1:].split("_", 1)[0].isdigit()]
        assert len(holonomic) == 16
        assert adapters[0] == "L10_Firewall"
        assert "L1_Quantum" in adapters
        assert "L16_Meta" in adapters
        assert registry.get("adapter", "L1_Quantum").__name__ == "L1_QuantumAdapter"
    finally:
        registry.clear("adapter")
        sys.modules.pop(module_name, None)
        importlib.import_module(module_name)


def test_namespaces(reg):
    @reg.register("neuron", "A")
    class A:
        pass

    @reg.register("layer", "B")
    class B:
        pass

    assert reg.namespaces() == ["layer", "neuron"]


def test_clear_namespace(reg):
    @reg.register("neuron", "X")
    class X:
        pass

    reg.clear("neuron")
    assert reg.list("neuron") == []


def test_clear_all(reg):
    @reg.register("neuron", "X")
    class X:
        pass

    reg.clear()
    assert reg.namespaces() == []


def test_singleton_import():
    from sc_neurocore.utils.registry import registry

    assert isinstance(registry, ComponentRegistry)

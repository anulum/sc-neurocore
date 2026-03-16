# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for the component registry

"""Tests for the component registry."""

from __future__ import annotations

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

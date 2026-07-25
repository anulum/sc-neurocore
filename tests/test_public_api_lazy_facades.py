# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Root and layer lazy-facade contracts

"""Verify advertised public symbols and lazy package facades."""

import sys
from types import ModuleType
from typing import cast

import pytest

import sc_neurocore


def test_all_symbols_importable() -> None:
    """Every advertised root package symbol resolves through the lazy facade."""
    for name in sc_neurocore.__all__:
        assert hasattr(sc_neurocore, name), f"Missing export: {name}"


def test_unknown_lazy_symbol_raises_attribute_error() -> None:
    """Unknown root-package attributes fail with the standard attribute error."""
    missing_name = "not_a_public_symbol"
    with pytest.raises(AttributeError, match="has no attribute 'not_a_public_symbol'"):
        getattr(sc_neurocore, missing_name)


def test_lazy_submodule_resolves_after_cache_eviction(monkeypatch: pytest.MonkeyPatch) -> None:
    """Public lazy submodules resolve through the package facade after eviction."""
    module_name = "sc_neurocore.datasets"
    monkeypatch.delitem(sc_neurocore.__dict__, "datasets", raising=False)
    monkeypatch.delitem(sys.modules, module_name, raising=False)

    resolved = sc_neurocore.datasets

    assert isinstance(resolved, ModuleType)
    assert resolved is sys.modules[module_name]
    assert sc_neurocore.__dict__["datasets"] is resolved
    assert "load_shd" in cast(list[str], resolved.__dict__["__all__"])


def test_dir_lists_lazy_public_api() -> None:
    """The package directory includes lazily exposed public API names."""
    public_names = dir(sc_neurocore)
    assert "StochasticLIFNeuron" in public_names
    assert "BitstreamEncoder" in public_names
    assert "JaxSCDenseLayer" in public_names
    assert "not_a_public_symbol" not in public_names


def test_jax_dense_layer_resolves_from_public_facades() -> None:
    """The optional JAX dense layer class resolves without constructing JAX state."""
    from sc_neurocore import layers
    from sc_neurocore.layers.jax_dense_layer import JaxSCDenseLayer

    assert sc_neurocore.JaxSCDenseLayer is JaxSCDenseLayer
    assert layers.JaxSCDenseLayer is JaxSCDenseLayer
    assert "JaxSCDenseLayer" in layers.__all__


def test_layers_lazy_facade_lists_exports_and_rejects_unknown_symbols() -> None:
    """The layer package lazy facade exposes directory entries and standard errors."""
    from sc_neurocore import layers

    missing_name = "not_a_layer_symbol"
    assert "JaxSCDenseLayer" in dir(layers)
    with pytest.raises(AttributeError, match=missing_name):
        getattr(layers, missing_name)

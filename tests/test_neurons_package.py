# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — neuron package lazy facade tests

"""Tests for the public `sc_neurocore.neurons` lazy package facade."""

from __future__ import annotations

import pytest

from sc_neurocore import neurons
from sc_neurocore.neurons import models


def test_load_rust_map_respects_disable_flag_and_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The optional Rust dispatch cache honours opt-out and cache reuse."""

    monkeypatch.setenv("SC_NEUROCORE_NO_RUST", "1")
    monkeypatch.setattr(neurons, "_rust_map", None)

    neurons._load_rust_map()

    assert neurons._rust_map == {}

    sentinel: dict[str, type] = {"AdExNeuron": models.AdExNeuron}
    monkeypatch.setattr(neurons, "_rust_map", sentinel)
    monkeypatch.delenv("SC_NEUROCORE_NO_RUST", raising=False)

    neurons._load_rust_map()

    assert neurons._rust_map is sentinel


def test_neurons_lazy_fallback_loads_python_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Package-level model access falls back to the pure-Python registry."""

    monkeypatch.setenv("SC_NEUROCORE_NO_RUST", "1")
    monkeypatch.setattr(neurons, "_rust_map", None)
    model_name = "AdExNeuron"
    monkeypatch.delitem(vars(neurons), model_name, raising=False)

    loaded = getattr(neurons, model_name)

    assert loaded is models.AdExNeuron
    assert vars(neurons)[model_name] is models.AdExNeuron


def test_neurons_lazy_unknown_symbol_raises_attribute_error() -> None:
    """Unknown package-level neuron names fail with the standard error."""

    missing_name = "UnknownNeuron"
    with pytest.raises(AttributeError, match="UnknownNeuron"):
        getattr(neurons, missing_name)

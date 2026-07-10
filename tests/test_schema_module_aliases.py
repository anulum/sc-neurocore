# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — schema_module_aliases tests

"""Tests for the schema ↔ module ↔ class alias registry."""

from __future__ import annotations

from sc_neurocore.neurons.models import _CLASS_TO_MODULE
from sc_neurocore.neurons.schema_module_aliases import (
    SCHEMA_SOURCE_ALIASES,
    SCHEMA_TO_CLASS,
    class_for_schema,
    module_for_schema,
    resolve_schema_join,
    schema_for_module,
)


def test_historical_stem_aliases_round_trip() -> None:
    """exp_if / resonate_fire keep the documented module joins."""
    assert module_for_schema("exp_if") == "expif"
    assert schema_for_module("expif") == "exp_if"
    assert module_for_schema("resonate_fire") == "resonate_and_fire"
    assert schema_for_module("resonate_and_fire") == "resonate_fire"
    assert SCHEMA_SOURCE_ALIASES["expif"] == "exp_if"
    assert SCHEMA_SOURCE_ALIASES["resonate_and_fire"] == "resonate_fire"


def test_lif_schema_joins_lapicque_not_stochastic_lif() -> None:
    """Schema lif is physiological/Lapicque lineage; SC flagship is separate."""
    module, class_name = resolve_schema_join("lif")
    assert module == "lapicque"
    assert class_name == "LapicqueNeuron"
    assert class_for_schema("lif") == "LapicqueNeuron"
    assert "StochasticLIFNeuron" in _CLASS_TO_MODULE
    assert _CLASS_TO_MODULE["StochasticLIFNeuron"] == "stochastic_lif"
    assert class_for_schema("lif") != "StochasticLIFNeuron"


def test_schema_to_class_entries_exist_in_registry() -> None:
    """Every mapped class_name is a registered catalogue model."""
    missing = [
        class_name for class_name in SCHEMA_TO_CLASS.values() if class_name not in _CLASS_TO_MODULE
    ]
    assert missing == [], f"alias points at unregistered classes: {missing}"


def test_identity_schema_module_when_unmapped() -> None:
    """Unknown stems fall back to identity without inventing joins."""
    assert module_for_schema("totally_unknown_model_xyz") == "totally_unknown_model_xyz"
    assert schema_for_module("totally_unknown_model_xyz") == "totally_unknown_model_xyz"
    assert class_for_schema("totally_unknown_model_xyz") is None

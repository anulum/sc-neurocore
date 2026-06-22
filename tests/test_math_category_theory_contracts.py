# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for category-theory bridge contracts

"""Contracts for stochastic, quantum, and biological category mappings."""

from __future__ import annotations

import numpy as np
import pytest

from sc_neurocore.math.category_theory import CategoryObject, CategoryTheoryBridge, Morphism


def test_stochastic_quantum_biological_functors_preserve_domains() -> None:
    bitstream = np.random.randint(0, 2, 64).astype(np.uint8)
    state_vector = CategoryTheoryBridge.stochastic_to_quantum(bitstream)
    biological = CategoryTheoryBridge.quantum_to_bio(np.array([1 / np.sqrt(2), 1 / np.sqrt(2)]))
    encoded = CategoryTheoryBridge.bio_to_stochastic(0.7, length=32)

    assert isinstance(state_vector, np.ndarray)
    assert isinstance(biological, float)
    assert encoded.shape == (32,)


def test_morphism_wraps_transformed_category_object() -> None:
    obj = CategoryObject(data=np.array([1.0, 2.0]), domain="stochastic")
    morphism = Morphism(func=lambda values: values * 2.0, name="double")

    result = morphism(obj)

    assert result.data[0] == pytest.approx(2.0)
    assert result.domain == "double"


def test_unknown_functor_is_rejected() -> None:
    with pytest.raises(ValueError):
        CategoryTheoryBridge().get_functor("Unknown", "Bio")


def test_each_named_functor_resolves_to_its_morphism() -> None:
    bridge = CategoryTheoryBridge()
    pairs = [
        ("Stochastic", "Quantum", "Sto->Quant"),
        ("Quantum", "Bio", "Quant->Bio"),
        ("Bio", "Stochastic", "Bio->Sto"),
    ]
    for source, target, label in pairs:
        morphism = bridge.get_functor(source, target)
        assert isinstance(morphism, Morphism)
        assert label in morphism.name

# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (every_model) from former test_model_descriptor.py

from __future__ import annotations

from tests.model_descriptor_support import *  # noqa: F403

@pytest.mark.parametrize("class_name", sorted(_CLASS_TO_MODULE))
def test_every_model_generates_a_valid_descriptor(class_name: str) -> None:
    """Gate: every registered model yields a schema-valid descriptor whose
    parameter and state names are all real fields of the model (no invented
    names)."""

    descriptor = generate_descriptor(class_name)
    assert descriptor.class_name == class_name
    declared = {p.name for p in descriptor.parameters} | {s.name for s in descriptor.state}

    module = importlib.import_module(f"sc_neurocore.neurons.models.{_CLASS_TO_MODULE[class_name]}")
    cls = getattr(module, class_name)
    if dataclasses.is_dataclass(cls):
        real_fields = {f.name for f in dataclasses.fields(cls)}
    else:
        real_fields = set(inspect.signature(cls).parameters)
    # A public read-only property is part of the model's real state surface even
    # when its storage is private (for example the canonical LFSR ``rng_state``).
    real_fields.update(
        name
        for base in cls.__mro__
        for name, member in vars(base).items()
        if isinstance(member, property)
    )
    # The synthetic fallback state "v" is allowed when a model declares none.
    invented = declared - real_fields - {"v"}
    assert invented == set(), f"{class_name}: descriptor names not in the model: {sorted(invented)}"



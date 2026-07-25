# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Schema validator error rendering tests

"""Validate stable human-readable schema error representations."""

from __future__ import annotations

from sc_neurocore.neurons.schema_validator import SchemaError


def test_schema_error_repr_includes_section_prefix() -> None:
    """SchemaError.__repr__ prefixes the section in brackets and upper-cases the level."""
    assert repr(SchemaError("error", "boom", "metadata")) == "ERROR: [metadata] boom"
    assert repr(SchemaError("warning", "soft")) == "WARNING: soft"

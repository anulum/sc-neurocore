# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — LIF co-simulation references

"""LIF schema precision reference contracts."""

from __future__ import annotations

from typing import Mapping, cast

from sc_neurocore.neurons.universal_dsl import UniversalNeuron


def _lif_schema_precision_values() -> dict[str, float]:
    """Return LIF schema values checked by the public precision CLI."""
    schema = UniversalNeuron.from_schema("lif").schema
    parameters = cast(Mapping[str, float], schema.get("parameters", {}))
    state = cast(Mapping[str, float], schema.get("state", {}))
    return {**parameters, **state}

# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Schema stem ↔ module / class alias registry

"""Canonical aliases between schema-DSL stems, model modules, and class names.

Schema files live at ``model_schemas/<stem>.toml``. Hand classes live under
``neurons/models/<module>.py`` (or a thin re-export). Several stems historically
diverged from module names (``exp_if`` vs ``expif``, ``resonate_fire`` vs
``resonate_and_fire``). This module is the single join table so schema-gap
reports, readiness indexing, Studio, and cosim harnesses do not invent
conflicting fuzzy matches.

The public SC flagship :class:`~sc_neurocore.neurons.stochastic_lif.StochasticLIFNeuron`
is **not** the same artefact as schema ``lif`` (physiological / Lapicque-lineage
DSL). Schema ``lif`` joins to :class:`~sc_neurocore.neurons.models.lapicque.LapicqueNeuron`
via the shared LIF product line; StochasticLIF is the normalised SC-oriented
API and has its own descriptor class.
"""

from __future__ import annotations

# Python model module stem (under neurons/models/) → schema-DSL stem.
# Only entries that **differ** need listing; identity maps are implicit.
MODULE_TO_SCHEMA: dict[str, str] = {
    "expif": "exp_if",
    "resonate_and_fire": "resonate_fire",
    "lapicque": "lapicque",  # explicit: schema "lif" is also LIF-lineage (see SCHEMA_TO_CLASS)
}

# Schema-DSL stem → preferred models/ module stem for hand-class join.
SCHEMA_TO_MODULE: dict[str, str] = {
    "exp_if": "expif",
    "resonate_fire": "resonate_and_fire",
    "lif": "lapicque",
    "lapicque": "lapicque",
    "izhikevich": "izhikevich2007",  # 2003 schema joins 2007 class until a distinct 2003 class exists
}

# Schema-DSL stem → public Python class_name used by descriptors / Studio.
SCHEMA_TO_CLASS: dict[str, str] = {
    "adex": "AdExNeuron",
    "connor_stevens": "ConnorStevensNeuron",
    "dpi_neuron": "DPINeuron",
    "exp_if": "ExpIFNeuron",
    "fitzhugh_nagumo": "FitzHughNagumoNeuron",
    "glif": "GLIFNeuron",
    "hindmarsh_rose": "HindmarshRoseNeuron",
    "hodgkin_huxley": "HodgkinHuxleyNeuron",
    "iqif": "IntegerQIFNeuron",
    "izhikevich": "Izhikevich2007Neuron",
    "izhikevich2007": "Izhikevich2007Neuron",
    "lapicque": "LapicqueNeuron",
    "lif": "LapicqueNeuron",
    "mckean": "McKeanNeuron",
    "mcculloch_pitts": "McCullochPittsNeuron",
    "mihalas_niebur": "MihalasNieburNeuron",
    "morris_lecar": "MorrisLecarNeuron",
    "perfect_integrator": "PerfectIntegratorNeuron",
    "quadratic_if": "QuadraticIFNeuron",
    "resonate_fire": "ResonateAndFireNeuron",
    "rulkov_map": "RulkovMapNeuron",
    "sc_upward_crossing_rulkov_map": "SCUpwardCrossingRulkovMapNeuron",
    "theta": "ThetaNeuron",
    "wang_buzsaki": "WangBuzsakiNeuron",
    "escape_rate": "EscapeRateNeuron",
    "poisson": "PoissonNeuron",
}

# Source-module stems that are aliases of another schema stem (schema_gap report).
# Key = models/*.py stem, value = schema stem when they differ.
SCHEMA_SOURCE_ALIASES: dict[str, str] = {
    "expif": "exp_if",
    "resonate_and_fire": "resonate_fire",
}


def schema_for_module(module: str) -> str:
    """Return the schema stem for a models/ module stem (identity if unmapped)."""
    return MODULE_TO_SCHEMA.get(module, SCHEMA_SOURCE_ALIASES.get(module, module))


def module_for_schema(schema: str) -> str:
    """Return the models/ module stem for a schema stem (identity if unmapped)."""
    return SCHEMA_TO_MODULE.get(schema, schema)


def class_for_schema(schema: str) -> str | None:
    """Return the descriptor class name for a schema stem, if known."""
    return SCHEMA_TO_CLASS.get(schema)


def resolve_schema_join(schema: str) -> tuple[str, str | None]:
    """Return ``(module, class_name_or_none)`` for a schema stem."""
    return module_for_schema(schema), class_for_schema(schema)


__all__ = [
    "MODULE_TO_SCHEMA",
    "SCHEMA_SOURCE_ALIASES",
    "SCHEMA_TO_CLASS",
    "SCHEMA_TO_MODULE",
    "class_for_schema",
    "module_for_schema",
    "resolve_schema_join",
    "schema_for_module",
]

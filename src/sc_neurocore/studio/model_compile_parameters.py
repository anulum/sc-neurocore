# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Catalogue parameters translated into a model's schema

"""Translate a catalogue model's parameters into the overrides of its canonical schema.

The Studio shows and sends a model's catalogue parameters: the names of its
Python class. Its RTL is compiled from the model's canonical schema, and nine
schemas name some parameters differently or hold them in another form. Each
translation here states what the class and the schema already state:

- a rename, where only the name differs (AdEx's ``b`` is the schema's
  ``b_adapt``);
- a derivation, where the schema holds values the class computes (Jansen-Rit's
  four couplings from ``c``; the Amari field's sampled kernel weights from its
  four kernel parameters and ``dx``);
- the seed, which the stochastic classes and their schemas both feed to the
  canonical sixteen-bit LFSR.

A value the schema cannot take is refused in words rather than dropped or
approximated: the RTL must be the neuron the reader configured.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from sc_neurocore.neurons.seed_domain import LFSR16_DOMAIN, SeedOutOfDomain, check_seed
from sc_neurocore.refusals import AuthoredRefusal


class CatalogueParametersRefused(AuthoredRefusal):
    """A catalogue parameter the schema cannot take, in words written for the caller."""


@dataclass(frozen=True, slots=True)
class SchemaParameters:
    """What a catalogue request becomes in the schema.

    Attributes
    ----------
    parameters : dict[str, float]
        Schema parameter overrides.
    rng_seed : int or None
        The seed of the schema's LFSR16, for a model whose catalogue carries one.
    """

    parameters: dict[str, float]
    rng_seed: int | None


#: The conductance-model names: the classes use lower case, the schemas the textbook capitals.
_CONDUCTANCE_NAMES: Mapping[str, str] = {
    "c_m": "C",
    "g_na": "g_Na",
    "g_k": "g_K",
    "g_l": "g_L",
    "e_na": "E_Na",
    "e_k": "E_K",
    "e_l": "E_L",
}

#: Catalogue names that differ from the schema's only in name, per catalogue model.
RENAMED: Mapping[str, Mapping[str, str]] = {
    "AdExNeuron": {"b": "b_adapt", "c_m": "C", "delta_t": "delta_T"},
    "BrunelWangNeuron": {"C_m": "c_m"},
    "HodgkinHuxleyNeuron": _CONDUCTANCE_NAMES,
    "ResonateAndFireNeuron": {"threshold": "v_threshold"},
    "WangBuzsakiNeuron": _CONDUCTANCE_NAMES,
}

#: Catalogue models whose ``seed`` is the seed of the schema's canonical LFSR16.
SEEDED: frozenset[str] = frozenset({"EscapeRateNeuron", "PoissonNeuron"})

Derivation = Callable[
    [str, dict[str, float], Mapping[str, float], Mapping[str, object]], dict[str, float]
]


def _jansen_rit(
    model_name: str,
    params: dict[str, float],
    defaults: Mapping[str, float],
    schema: Mapping[str, object],
) -> dict[str, float]:
    """Derive the four couplings from the base connectivity ``c``: C1 = c, C2 = 0.8 c, C3 = C4 = 0.25 c.

    Parameters
    ----------
    model_name : str
        The catalogue model, for messages.
    params : dict[str, float]
        The request's parameters; ``c`` is taken out.
    defaults : Mapping[str, float]
        The catalogue's defaults (unused: ``c`` alone sets the couplings).
    schema : Mapping[str, object]
        The canonical schema (unused).

    Returns
    -------
    dict[str, float]
        The couplings, when the request carries ``c``.
    """
    if "c" not in params:
        return {}
    c = params.pop("c")
    return {"c1": c, "c2": 0.8 * c, "c3": 0.25 * c, "c4": 0.25 * c}


_AMARI_KERNEL = ("a_exc", "a_width", "b_inh", "b_width")


def _amari_field(
    model_name: str,
    params: dict[str, float],
    defaults: Mapping[str, float],
    schema: Mapping[str, object],
) -> dict[str, float]:
    """Sample the kernel ``w(d) = a_exc exp(-a_width d) - b_inh exp(-b_width d)`` at d = 0, dx and 2 dx.

    The schema is the field on a fixed ring of sites (its ``site_count``), on
    which the distances between sites are 0, ``dx`` and ``2 dx``; the class
    samples its kernel the same way. ``n`` must be that number of sites, and
    is taken out; ``dx`` stays, as the schema's own parameter.

    A request that carries ``n`` or a kernel parameter is in catalogue names,
    which the schema does not have, so the weights are derived at the
    request's ``dx`` even when it carries no kernel parameter: the schema
    uses ``dx`` both as its integration weight and as the kernel's sampling
    distance, and keeping weights sampled at another ``dx`` would compile a
    different field.

    Parameters
    ----------
    model_name : str
        The catalogue model, for messages.
    params : dict[str, float]
        The request's parameters; ``n`` and the kernel's four are taken out.
    defaults : Mapping[str, float]
        The catalogue's defaults, for kernel parameters the request leaves out.
    schema : Mapping[str, object]
        The canonical schema, which states its number of sites.

    Returns
    -------
    dict[str, float]
        The sampled weights, when the request carries ``n`` or a kernel
        parameter.

    Raises
    ------
    CatalogueParametersRefused
        When ``n`` is not the schema's number of sites.
    """
    extensions = schema.get("extensions")
    sites = extensions.get("site_count") if isinstance(extensions, Mapping) else None
    if not isinstance(sites, int):
        raise ValueError(f"Schema of {model_name!r} states no integer site_count.")
    in_catalogue_names = "n" in params or any(name in params for name in _AMARI_KERNEL)
    if "n" in params:
        n = params.pop("n")
        if n != sites:
            raise CatalogueParametersRefused(
                f"{model_name} compiles to the schema's fixed field of {sites} sites; "
                f"n = {n:g} cannot be compiled. Set n to {sites}."
            )
    given = {name: params.pop(name) for name in _AMARI_KERNEL if name in params}
    if not in_catalogue_names:
        return {}
    kernel = {name: given.get(name, _default(model_name, defaults, name)) for name in _AMARI_KERNEL}
    dx = params.get("dx", _default(model_name, defaults, "dx"))
    return {
        f"w{k}": kernel["a_exc"] * math.exp(-kernel["a_width"] * k * dx)
        - kernel["b_inh"] * math.exp(-kernel["b_width"] * k * dx)
        for k in range(3)
    }


def _default(model_name: str, defaults: Mapping[str, float], name: str) -> float:
    """Return a catalogue default a derivation needs, or name the server's missing record.

    Parameters
    ----------
    model_name : str
        The catalogue model, for messages.
    defaults : Mapping[str, float]
        The catalogue's defaults.
    name : str
        The parameter needed.

    Returns
    -------
    float
        Its default.

    Raises
    ------
    ValueError
        When the catalogue states none, a fault of the server's catalogue.
    """
    if name not in defaults:
        raise ValueError(f"Studio model {model_name!r} has no catalogue default for {name!r}.")
    return defaults[name]


#: Catalogue models whose schema holds values the class derives from its own parameters.
DERIVED: Mapping[str, Derivation] = {
    "AmariNeuralField": _amari_field,
    "JansenRitUnit": _jansen_rit,
}


def _seed(model_name: str, value: float) -> int:
    """Return a catalogue seed as the LFSR16 seed, or refuse it in words.

    Parameters
    ----------
    model_name : str
        The catalogue model, for messages.
    value : float
        The seed as the request carries it.

    Returns
    -------
    int
        The seed.

    Raises
    ------
    CatalogueParametersRefused
        When it is not a whole number, or not in the LFSR16 state space.
    """
    if not float(value).is_integer():
        raise CatalogueParametersRefused(
            f"{model_name}'s seed must be a whole number; received {value:g}."
        )
    try:
        return check_seed(model_name, int(value), LFSR16_DOMAIN)
    except SeedOutOfDomain as refused:
        raise CatalogueParametersRefused(str(refused)) from None


def to_schema_parameters(
    model_name: str,
    params: Mapping[str, float],
    *,
    catalogue_defaults: Mapping[str, float],
    schema: Mapping[str, object],
) -> SchemaParameters:
    """Translate a request's catalogue parameters into its schema's overrides.

    Names the schema already uses pass through, as before; a name neither side
    knows is left for the caller's unknown-parameter check.

    Parameters
    ----------
    model_name : str
        The catalogue model.
    params : Mapping[str, float]
        The request's parameters.
    catalogue_defaults : Mapping[str, float]
        The catalogue's default for each of the model's parameters.
    schema : Mapping[str, object]
        The model's canonical schema.

    Returns
    -------
    SchemaParameters
        The schema overrides and the LFSR16 seed.

    Raises
    ------
    CatalogueParametersRefused
        When a request names one parameter twice, by its catalogue and its
        schema name, or carries a value the schema cannot take.
    """
    remaining = dict(params)
    rng_seed = None
    if model_name in SEEDED and "seed" in remaining:
        rng_seed = _seed(model_name, remaining.pop("seed"))
    translated: dict[str, float] = {}
    for catalogue_name, schema_name in RENAMED.get(model_name, {}).items():
        if catalogue_name not in remaining:
            continue
        if schema_name in remaining:
            raise CatalogueParametersRefused(
                f"{model_name} received both {catalogue_name!r} and its schema name "
                f"{schema_name!r}; send one of them."
            )
        translated[schema_name] = remaining.pop(catalogue_name)
    derive = DERIVED.get(model_name)
    if derive is not None:
        derived = derive(model_name, remaining, catalogue_defaults, schema)
        given = sorted(set(derived) & set(remaining))
        if given:
            raise CatalogueParametersRefused(
                f"{model_name} received {', '.join(repr(name) for name in given)}, which it derives "
                "from its catalogue parameters; send those instead."
            )
        translated.update(derived)
    return SchemaParameters(parameters={**remaining, **translated}, rng_seed=rng_seed)

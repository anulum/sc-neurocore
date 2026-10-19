# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Distinct, reproducible seeds for the neurons of one population

"""Give every neuron in a population its own random stream.

A population built from one seed gave every neuron the same stream, so a
group of stochastic neurons produced one spike train repeated *n* times.
Measured before this module existed: eight ``EscapeRateNeuron`` driven at
60 units for 60 steps produced eight identical trains of nine spikes. Every
statistic taken over such a population — variance, synchrony, the fluctuation
of the population rate — described one neuron and reported it as a group.

Seeds are derived rather than drawn, so the population stays reproducible: the
same base seed, count, domain and aliases give the same seeds on any machine,
in any order, in any process. They are also derived **distinct** in the state
the generator starts from, because two neurons sharing a stream is the defect
this module exists to remove, and a hash alone collides long before the domain
is exhausted.

Each model declares the seeds it accepts (``SEED_DOMAIN``, see
:mod:`sc_neurocore.neurons.seed_domain`) and, where two request seeds start the
same generator state, which ones alias (``SEED_ALIASES``): the 16-bit LFSR
models accept ``[0, 65535]`` but run seed ``0`` as the fallback ``0xACE1``, so
they hold 65535 distinct states. Seeds are derived in the model's domain, and
a seed whose state another neuron already starts from is skipped, so the
capacity of a population is the number of distinct states, not of request
seeds. A model that declares nothing is given :data:`UNIVERSAL_SEED_DOMAIN`,
``[1, 65535]``, the narrowest domain in this build. Distinct seeds give
distinct, reproducible streams; they are not a proof of statistical
independence between the streams.
"""

from __future__ import annotations

from collections.abc import Mapping
from hashlib import blake2b

from sc_neurocore.neurons.seed_domain import UNIVERSAL_SEED_DOMAIN, SeedDomain

LFSR16_SEED_DOMAIN = 65535
"""Largest seed the narrowest domain in this build accepts."""


def derive_population_seeds(
    base_seed: int,
    count: int,
    domain: SeedDomain = UNIVERSAL_SEED_DOMAIN,
    aliases: Mapping[int, int] | None = None,
) -> list[int]:
    """Derive one seed per neuron, each starting a distinct state, reproducibly.

    Parameters
    ----------
    base_seed : int
        The seed the population was asked for. It is the only entropy used, so
        two populations built with the same base seed and count are identical.
    count : int
        How many neurons need a seed.
    domain : tuple of int, default UNIVERSAL_SEED_DOMAIN
        The inclusive ``(low, high)`` seeds the model accepts (its
        ``SEED_DOMAIN``); every derived seed lies in it.
    aliases : mapping of int to int, optional
        Request seeds the model runs as another seed (its ``SEED_ALIASES``),
        such as LFSR16's ``{0: 0xACE1}``. Two seeds that start the same state
        are never both handed out.

    Returns
    -------
    list of int
        ``count`` seeds in ``domain``, in neuron order, whose starting states
        are pairwise distinct. The seeds of the first ``k`` neurons do not
        depend on ``count``.

    Raises
    ------
    ValueError
        If ``count`` is negative, or exceeds the number of distinct states the
        domain holds once aliases are merged (65535 for the LFSR16 models).
        Refusing is deliberate: the alternative is handing two neurons the same
        stream, which is the defect this exists to remove.
    """
    if count < 0:
        raise ValueError("count must not be negative")
    low, high = domain
    size = high - low + 1
    alias = {
        request: start
        for request, start in (aliases or {}).items()
        if low <= request <= high and request != start
    }
    capacity = size - sum(1 for start in alias.values() if low <= start <= high)
    if count > capacity:
        raise ValueError(
            f"a population of {count} seeded neurons cannot be given distinct seeds: "
            f"the model's declared domain [{low}, {high}] holds {capacity} distinct "
            "generator states. Split the population, or use a model with a wider "
            "declared domain."
        )
    seeds: list[int] = []
    used: set[int] = set()
    for index in range(count):
        candidate = low + _derived_seed(base_seed, index) % size
        while alias.get(candidate, candidate) in used:
            # Walk forward through the domain past every seed whose starting
            # state is taken. Terminates because fewer states are taken than
            # the domain holds, and it is deterministic, so the same base seed,
            # domain and aliases always produce the same list.
            candidate = low + (candidate - low + 1) % size
        used.add(alias.get(candidate, candidate))
        seeds.append(candidate)
    return seeds


def _derived_seed(base_seed: int, index: int) -> int:
    """Return the seed this base and index hash to, before collision walking.

    Parameters
    ----------
    base_seed : int
        The population's base seed.
    index : int
        The neuron's position in the population.

    Returns
    -------
    int
        A non-negative 64-bit value the caller reduces into the model's
        declared domain. The digest is taken over the decimal text of both
        values so the derivation does not depend on the width or byte order of
        a machine integer.
    """
    digest = blake2b(f"{base_seed}:{index}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big")


__all__ = ["LFSR16_SEED_DOMAIN", "derive_population_seeds"]

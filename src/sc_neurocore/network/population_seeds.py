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
same base seed and count give the same seeds on any machine, in any order, in
any process. They are also derived **distinct**, because two neurons sharing a
stream is the defect this module exists to remove, and a hash alone collides
long before the domain is exhausted.

The domain is ``[1, 65535]``. That is the narrowest seed domain in this build
— the canonical LFSR16 recurrence, which rejects a seed outside it and treats
zero as a request for the documented fallback — and it is inside the domain of
every numpy-seeded model as well. Per-model seed domains are not declared
anywhere yet, so this stays inside the intersection of the domains the build is
known to have rather than assuming a wider one.
"""

from __future__ import annotations

from hashlib import blake2b

LFSR16_SEED_DOMAIN = 65535
"""Largest seed the narrowest domain in this build accepts."""


def derive_population_seeds(base_seed: int, count: int) -> list[int]:
    """Derive one distinct seed per neuron, reproducibly.

    Parameters
    ----------
    base_seed : int
        The seed the population was asked for. It is the only entropy used, so
        two populations built with the same base seed and count are identical.
    count : int
        How many neurons need a seed.

    Returns
    -------
    list of int
        ``count`` distinct seeds in ``[1, 65535]``, in neuron order.

    Raises
    ------
    ValueError
        If ``count`` is negative, or exceeds the number of distinct seeds the
        domain holds. Refusing is deliberate: the alternative is handing two
        neurons the same stream, which is the defect this exists to remove.
    """
    if count < 0:
        raise ValueError("count must not be negative")
    if count > LFSR16_SEED_DOMAIN:
        raise ValueError(
            f"a population of {count} seeded neurons cannot be given distinct seeds: "
            f"the domain holds {LFSR16_SEED_DOMAIN}. Split the population, or use a "
            "model whose seed domain is declared and wider."
        )
    seeds: list[int] = []
    used: set[int] = set()
    for index in range(count):
        candidate = _derived_seed(base_seed, index)
        while candidate in used:
            # Walk forward through the domain. Terminates because the number of
            # seeds already taken is smaller than the domain, and it is
            # deterministic, so the same base seed and count always produce the
            # same list.
            candidate = candidate % LFSR16_SEED_DOMAIN + 1
        used.add(candidate)
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
        A seed in ``[1, 65535]``. The digest is taken over the decimal text of
        both values so the derivation does not depend on the width or byte
        order of a machine integer.
    """
    digest = blake2b(f"{base_seed}:{index}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big") % LFSR16_SEED_DOMAIN + 1


__all__ = ["LFSR16_SEED_DOMAIN", "derive_population_seeds"]

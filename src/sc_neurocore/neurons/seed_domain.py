# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — The seeds a model accepts

"""Which seeds a model accepts, declared rather than discovered on failure.

A seeded model's constructor already knows its domain: a 16-bit LFSR takes
``0..65535``, a NumPy-backed model takes any non-negative integer, and one
model excludes zero because zero is its generator's dead state. Nothing
published that knowledge, so a caller learned a domain by being refused — the
Studio reported an out-of-range seed as an invalid model input from the
constructor, after the run had been accepted, rather than as a request the
experiment contract could refuse with the domain named.

Two things followed from the silence. Every fresh seed was drawn from the
narrowest domain any model has, sixteen bits, because that is the only value
safe for all of them; and a population deriving a distinct seed per neuron had
to do the same, so a model with a 63-bit domain received seeds from a 16-bit
one.

A model declares its domain with a ``SEED_DOMAIN`` class attribute. It is a
statement of what that constructor already enforces, made per identity, not a
new rule imposed on it.
"""

from __future__ import annotations

from typing import Final

#: The domain every seeded model in the catalogue accepts, used where the
#: model is unknown or declares nothing. It is the narrowest declared domain,
#: so a value drawn from it is safe everywhere and wasteful nowhere else.
UNIVERSAL_SEED_DOMAIN: Final = (1, 65535)

#: The 16-bit LFSR domain: the shared hardware generator's whole state space.
LFSR16_DOMAIN: Final = (0, 65535)

#: A NumPy-backed model's domain. `SeedSequence` accepts any non-negative
#: integer; the upper bound is the largest a signed 64-bit field carries, which
#: is what the request and receipt surfaces transport.
NUMPY_SEED_DOMAIN: Final = (0, 2**63 - 1)

#: The same domain with zero removed, for a generator whose zero state is dead.
NONZERO_SEED_DOMAIN: Final = (1, 2**63 - 1)

SeedDomain = tuple[int, int]


class SeedOutOfDomain(ValueError):
    """Raised when a seed falls outside the domain its model declares.

    Parameters
    ----------
    model : str
        The model the seed was meant for.
    seed : int
        The value offered.
    domain : tuple of int
        The inclusive ``(low, high)`` bounds the model declares.
    """

    def __init__(self, model: str, seed: int, domain: SeedDomain) -> None:
        low, high = domain
        super().__init__(f"{model} accepts a seed in [{low}, {high}]; received {seed}")
        self.model = model
        self.seed = seed
        self.domain = domain


def seed_domain(model: type[object] | None) -> SeedDomain:
    """Return the seeds a model accepts.

    Parameters
    ----------
    model : type or None
        The model class, or ``None`` when the caller has no class to ask.

    Returns
    -------
    tuple of int
        The inclusive ``(low, high)`` bounds the model declares, or
        :data:`UNIVERSAL_SEED_DOMAIN` when it declares none. An undeclared
        model is given the narrowest domain rather than the widest, because a
        seed that is too small is always accepted and one that is too large is
        refused by the constructor after the run was admitted.

    Examples
    --------
    >>> from sc_neurocore.neurons.models.poisson import PoissonNeuron
    >>> seed_domain(PoissonNeuron)
    (0, 65535)
    >>> seed_domain(None)
    (1, 65535)
    """
    declared = getattr(model, "SEED_DOMAIN", None) if model is not None else None
    if not isinstance(declared, tuple) or len(declared) != 2:
        return UNIVERSAL_SEED_DOMAIN
    low, high = declared
    if not isinstance(low, int) or not isinstance(high, int) or isinstance(low, bool):
        return UNIVERSAL_SEED_DOMAIN
    if isinstance(high, bool) or low > high:
        return UNIVERSAL_SEED_DOMAIN
    return (low, high)


def check_seed(model_name: str, seed: int, domain: SeedDomain) -> int:
    """Return *seed* when the domain admits it, or raise :class:`SeedOutOfDomain`."""
    low, high = domain
    if not isinstance(seed, int) or isinstance(seed, bool) or not low <= seed <= high:
        raise SeedOutOfDomain(model_name, seed, domain)
    return seed


__all__ = [
    "LFSR16_DOMAIN",
    "NONZERO_SEED_DOMAIN",
    "NUMPY_SEED_DOMAIN",
    "UNIVERSAL_SEED_DOMAIN",
    "SeedDomain",
    "SeedOutOfDomain",
    "check_seed",
    "seed_domain",
]

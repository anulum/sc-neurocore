# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Functor mapping between distinct computational domains

"""Category-theoretic functors mapping stochastic, quantum, and bio domains."""

from typing import Any, Callable, TypeVar, Generic
from dataclasses import dataclass
import numpy as np

T = TypeVar("T")
U = TypeVar("U")


@dataclass
class CategoryObject(Generic[T]):
    """Domain-tagged value transported between computational categories."""

    data: T
    domain: str


class Morphism:
    """Named structure-preserving map between two :class:`CategoryObject` domains."""

    def __init__(self, func: Callable[[Any], Any], name: str):
        self.func = func
        self.name = name

    def __call__(self, obj: CategoryObject[Any]) -> CategoryObject[Any]:
        """Apply the morphism, tagging the result with the morphism name."""
        return CategoryObject(data=self.func(obj.data), domain=self.name)


class CategoryTheoryBridge:
    """Functors mapping between the stochastic, quantum, and bio domains."""

    @staticmethod
    def stochastic_to_quantum(bitstream: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Map a bitstream probability ``p`` to the quantum amplitude pair."""
        p = np.mean(bitstream)
        # Quantum state |psi> = sqrt(p)|1> + sqrt(1-p)|0>
        alpha = np.sqrt(1 - p)
        beta = np.sqrt(p)
        return np.array([alpha, beta])

    @staticmethod
    def quantum_to_bio(state_vector: np.ndarray[Any, Any]) -> float:
        """Map quantum probability ``|beta|^2`` to a concentration in ``[0, 10]`` uM."""
        prob_1 = np.abs(state_vector[1]) ** 2
        concentration = prob_1 * 10.0
        return float(concentration)

    @staticmethod
    def bio_to_stochastic(concentration: float, length: int = 100) -> np.ndarray[Any, Any]:
        """Map a concentration to a Bernoulli bitstream of the given length."""
        p = np.clip(concentration / 10.0, 0, 1)
        rands = np.random.random(length)
        bitstream: np.ndarray[Any, Any] = (rands < p).astype(np.uint8)
        return bitstream

    def get_functor(self, source: str, target: str) -> Morphism:
        """Return the morphism mapping the ``source`` domain to ``target``."""
        if source == "Stochastic" and target == "Quantum":
            return Morphism(self.stochastic_to_quantum, "Functor: Sto->Quant")
        if source == "Quantum" and target == "Bio":
            return Morphism(self.quantum_to_bio, "Functor: Quant->Bio")
        if source == "Bio" and target == "Stochastic":
            return Morphism(self.bio_to_stochastic, "Functor: Bio->Sto")
        raise ValueError(f"No morphism from {source} to {target}")

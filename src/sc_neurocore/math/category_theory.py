# SPDX-License-Identifier: AGPL-3.0-or-later
from typing import Any, Callable, TypeVar, Generic
from dataclasses import dataclass
import numpy as np

T = TypeVar("T")
U = TypeVar("U")


@dataclass
class CategoryObject(Generic[T]):
    data: T
    domain: str


class Morphism:
    def __init__(self, func: Callable[[Any], Any], name: str):
        self.func = func
        self.name = name

    def __call__(self, obj: CategoryObject) -> CategoryObject:  # type: ignore
        return CategoryObject(data=self.func(obj.data), domain=self.name)


class CategoryTheoryBridge:
    """
    Functor mapping between distinct computational domains.
    Stochastic <-> Quantum <-> Bio
    """

    @staticmethod
    def stochastic_to_quantum(bitstream: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """
        Map bitstream probability p to quantum amplitude sqrt(p).
        """
        p = np.mean(bitstream)
        # Quantum state |psi> = sqrt(p)|1> + sqrt(1-p)|0>
        alpha = np.sqrt(1 - p)
        beta = np.sqrt(p)
        return np.array([alpha, beta])

    @staticmethod
    def quantum_to_bio(state_vector: np.ndarray[Any, Any]) -> float:
        """
        Map quantum probability |beta|^2 to concentration [0, 10] uM.
        """
        prob_1 = np.abs(state_vector[1]) ** 2
        concentration = prob_1 * 10.0
        return concentration  # type: ignore

    @staticmethod
    def bio_to_stochastic(concentration: float, length: int = 100) -> np.ndarray[Any, Any]:
        """
        Map concentration to bitstream.
        """
        p = np.clip(concentration / 10.0, 0, 1)
        rands = np.random.random(length)
        return (rands < p).astype(np.uint8)  # type: ignore

    def get_functor(self, source: str, target: str) -> Morphism:
        if source == "Stochastic" and target == "Quantum":
            return Morphism(self.stochastic_to_quantum, "Functor: Sto->Quant")
        if source == "Quantum" and target == "Bio":
            return Morphism(self.quantum_to_bio, "Functor: Quant->Bio")
        if source == "Bio" and target == "Stochastic":
            return Morphism(self.bio_to_stochastic, "Functor: Bio->Sto")
        raise ValueError(f"No morphism from {source} to {target}")

# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Hyper-Dimensional Computing (HDC/VSA) with

"""Hyper-Dimensional Computing (HDC/VSA) with hardware-accelerated bitstreams.

This module wraps the Rust ``BitStreamTensor`` to provide a Pythonic
HDC vector class with operator overloading:

*  ``*``  — bind   (XOR)
*  ``+``  — bundle (majority vote)
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING as _TYPE_CHECKING
from typing import Sequence

from sc_neurocore_engine.sc_neurocore_engine import (
    BitStreamTensor as _RustBitStreamTensor,
)

if _TYPE_CHECKING:
    from ._binding_types import BitStreamTensor as _TensorType

_seed_counter = 0
_seed_lock = threading.Lock()


def _next_seed() -> int:
    """Allocate the next process-local hypervector seed under the shared lock."""
    global _seed_counter
    with _seed_lock:
        _seed_counter += 1
        return _seed_counter


class HDCVector:
    """Hyper-Dimensional Computing vector (default 10 000-bit)."""

    __slots__ = ("_tensor",)

    def __init__(self, dimension: int = 10_000, *, seed: int | None = None) -> None:
        """Construct a native random binary hypervector.

        Parameters
        ----------
        dimension : int, optional
            Positive logical number of bits, default 10000.
        seed : int or None, optional
            Replayable native seed; None allocates a process-local seed.
        """
        self._tensor: _TensorType = _RustBitStreamTensor(
            dimension, seed if seed is not None else _next_seed()
        )

    # ── factories ─────────────────────────────────────────────────────

    @classmethod
    def _from_rust(cls, tensor: _RustBitStreamTensor) -> HDCVector:
        """Wrap an existing native tensor without copying its packed words.

        Parameters
        ----------
        tensor : BitStreamTensor
            Actual native tensor returned by an HDC operation.

        Returns
        -------
        HDCVector
            Facade retaining the supplied native tensor.
        """
        obj = object.__new__(cls)
        obj._tensor = tensor
        return obj

    @classmethod
    def from_packed(cls, data: list[int], length: int) -> HDCVector:
        """Construct a hypervector from native packed uint64 words.

        Parameters
        ----------
        data : list of int
            Packed words with zero unused high bits.
        length : int
            Positive logical bit count.

        Returns
        -------
        HDCVector
            Native tensor exposed through the vector facade.
        """
        return cls._from_rust(_RustBitStreamTensor.from_packed(data, length))

    # ── operators ─────────────────────────────────────────────────────

    def __mul__(self, other: HDCVector) -> HDCVector:
        """Return the hypervector bound to other by bitwise XOR."""
        return self.xor(other)

    def __add__(self, other: HDCVector) -> HDCVector:
        """Return a new vector formed by bundling these two hypervectors."""
        return HDCVector.bundle([self, other])

    # ── core ops ──────────────────────────────────────────────────────

    def xor(self, other: HDCVector) -> HDCVector:
        """Bind: returns a new vector that is the XOR of self and other."""
        return HDCVector._from_rust(self._tensor.xor(other._tensor))

    def permute(self, shift: int = 1) -> HDCVector:
        """Return a new permuted (cyclic right rotated) vector."""
        t = _RustBitStreamTensor.from_packed(self._tensor.data, self._tensor.length)
        t.rotate_right(shift)
        return HDCVector._from_rust(t)

    def similarity(self, other: HDCVector) -> float:
        """Cosine-like similarity (1.0 - hamming_distance)."""
        return 1.0 - self._tensor.hamming_distance(other._tensor)

    @staticmethod
    def bundle(vectors: Sequence[HDCVector]) -> HDCVector:
        """Majority-vote bundle across N vectors."""
        if len(vectors) == 0:
            raise ValueError("Cannot bundle zero vectors.")
        tensors = [v._tensor for v in vectors]
        return HDCVector._from_rust(_RustBitStreamTensor.bundle(tensors))

    # ── introspection ─────────────────────────────────────────────────

    @property
    def dimension(self) -> int:
        """Return the logical number of bits in the native tensor.

        Returns
        -------
        int
            Logical dimension, excluding unused packed high bits.
        """
        return self._tensor.length

    def popcount(self) -> int:
        """Count the set bits in the native tensor.

        Returns
        -------
        int
            Number of active bits within the logical dimension.
        """
        return self._tensor.popcount()

    def __len__(self) -> int:
        """Return the logical bit count of the wrapped native tensor."""
        return self._tensor.length

    def __repr__(self) -> str:
        """Describe the hypervector using its dimension and current set-bit count."""
        return f"HDCVector(dim={self.dimension}, popcount={self.popcount()})"

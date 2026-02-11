# CopyRight: (c) 1998-2026 Miroslav Sotek. All rights reserved.
# Contact us: www.anulum.li  protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AFFERO GENERAL PUBLIC LICENSE v3
# Commercial Licensing: Available

"""Hyper-Dimensional Computing (HDC/VSA) with hardware-accelerated bitstreams.

This module wraps the Rust ``BitStreamTensor`` to provide a Pythonic
HDC vector class with operator overloading:

*  ``*``  — bind   (XOR)
*  ``+``  — bundle (majority vote)
"""

from __future__ import annotations

import threading
from typing import Sequence

from sc_neurocore_engine.sc_neurocore_engine import (
    BitStreamTensor as _RustBitStreamTensor,
)

_seed_counter = 0
_seed_lock = threading.Lock()


def _next_seed() -> int:
    global _seed_counter
    with _seed_lock:
        _seed_counter += 1
        return _seed_counter


class HDCVector:
    """Hyper-Dimensional Computing vector (default 10 000-bit)."""

    __slots__ = ("_tensor",)

    def __init__(self, dimension: int = 10_000, *, seed: int | None = None):
        self._tensor = _RustBitStreamTensor(dimension, seed if seed is not None else _next_seed())

    # ── factories ─────────────────────────────────────────────────────

    @classmethod
    def _from_rust(cls, tensor: _RustBitStreamTensor) -> HDCVector:
        obj = object.__new__(cls)
        obj._tensor = tensor
        return obj

    @classmethod
    def from_packed(cls, data: list[int], length: int) -> HDCVector:
        """Create from pre-packed u64 words."""
        return cls._from_rust(_RustBitStreamTensor.from_packed(data, length))

    # ── operators ─────────────────────────────────────────────────────

    def __mul__(self, other: HDCVector) -> HDCVector:
        """Bind (XOR)."""
        return self.xor(other)

    def __add__(self, other: HDCVector) -> HDCVector:
        """Bundle (majority vote of two vectors)."""
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
        return self._tensor.length

    def popcount(self) -> int:
        return self._tensor.popcount()

    def __len__(self) -> int:
        return self._tensor.length

    def __repr__(self) -> str:
        return f"HDCVector(dim={self.dimension}, popcount={self.popcount()})"

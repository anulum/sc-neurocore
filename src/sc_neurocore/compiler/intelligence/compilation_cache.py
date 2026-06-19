# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Compilation cache

"""Memoized compilation result cache.

Avoids redundant recompilation of ODE systems when re-targeting or
running multi-target comparisons.
"""

from __future__ import annotations

from typing import Any


class CompilationCache:
    """Memoized compilation result cache.

    Keyed by ``(equations_hash, target, data_width, fraction)``.
    Avoids redundant recompilation when re-targeting.
    """

    def __init__(self) -> None:
        self._store: dict[str, dict[str, Any]] = {}
        self.hits: int = 0
        self.misses: int = 0

    def _key(
        self,
        equations: dict[str, str],
        target: str,
        data_width: int,
        fraction: int,
    ) -> str:
        import hashlib
        import json

        h = hashlib.sha256(
            json.dumps(
                {"eq": equations, "t": target, "w": data_width, "f": fraction},
                sort_keys=True,
            ).encode()
        ).hexdigest()[:16]
        return h

    def get(
        self,
        equations: dict[str, str],
        target: str,
        data_width: int = 16,
        fraction: int = 8,
    ) -> dict[str, Any] | None:
        """Look up a cached compilation result.

        Parameters
        ----------
        equations : dict[str, str]
            ODE equations.
        target : str
            Target profile name.
        data_width : int
            Fixed-point width.
        fraction : int
            Fractional bits.

        Returns
        -------
        dict or None
            Cached result if hit, None if miss.
        """
        key = self._key(equations, target, data_width, fraction)
        result = self._store.get(key)
        if result is not None:
            self.hits += 1
        else:
            self.misses += 1
        return result

    def put(
        self,
        equations: dict[str, str],
        target: str,
        data_width: int,
        fraction: int,
        result: dict[str, Any],
    ) -> None:
        """Store a compilation result in cache.

        Parameters
        ----------
        equations : dict[str, str]
            ODE equations.
        target : str
            Target profile name.
        data_width : int
            Fixed-point width.
        fraction : int
            Fractional bits.
        result : dict
            Compilation result to cache.
        """
        key = self._key(equations, target, data_width, fraction)
        self._store[key] = result

    @property
    def size(self) -> int:
        """Number of cached entries."""
        return len(self._store)

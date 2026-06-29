# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Canonical acceleration-backend dispatch order

"""Single source of truth for the polyglot backend dispatch order.

Compiled backends are listed fastest-measured-first; the pure-Python (or NumPy)
implementation is always the final, always-available floor. Kernel modules differ
only in which floor they fall back to — a hand-written Python loop (``"python"``)
or a vectorised NumPy path (``"numpy"``) — so the floor is appended per consumer
rather than baked into the shared order.

This replaces four copies of the same ``("rust", "mojo", "julia", "go", ...)``
literal. The data-driven selector in :mod:`sc_neurocore.accel.backend_selection`
reorders these names from measured benchmarks and falls back to this static order
when no measurement matches the running host.
"""

from __future__ import annotations

#: Compiled acceleration backends, fastest-measured-first (no floor).
ACCELERATORS: tuple[str, ...] = ("rust", "mojo", "julia", "go")


def with_floor(floor: str = "python") -> tuple[str, ...]:
    """Return :data:`ACCELERATORS` with ``floor`` appended as the always-available tier."""
    return (*ACCELERATORS, floor)


#: Default kernel dispatch order with the pure-Python floor.
FASTEST_FIRST_BACKENDS: tuple[str, ...] = with_floor("python")

__all__ = ["ACCELERATORS", "FASTEST_FIRST_BACKENDS", "with_floor"]

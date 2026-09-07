# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Built-lane gate for backend parity tests

"""Built-lane gate: skip a lane that was never built, never mask CI.

The Rust extension and the Julia bridge already have gates of this shape
(:mod:`tests.engine_requirement`, :mod:`tests.julia_requirement`). The Go and
Mojo lanes had none, because their absence is not a missing Python module but a
missing shared library beside the model's accelerator package. A backend-
parametrised test therefore did not skip on a checkout where those lanes had
never been built: it failed with ``<lane> backend is unavailable``, which reads
as a defect in the model rather than as an absent toolchain.

The gate asks the model's own accelerator module whether the lane loads, which
is the same question the dispatcher asks, so it cannot answer differently from
the code under test. A lane that is genuinely absent skips the parametrisation
with a named reason. ``SC_NEUROCORE_REQUIRE_GO=1`` and
``SC_NEUROCORE_REQUIRE_MOJO=1`` disable the skip entirely, and hosted CI exports
both alongside the engine and Julia switches it already sets — so a lane that
fails to build there is a hard failure and can never hide behind a green wall of
skips.

Everything after a healthy load stays a hard failure. This gate answers one
question only: was this lane built here at all.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from types import ModuleType

import pytest

#: The lanes this gate covers. The Rust extension and the Julia bridge are
#: gated by their own modules, which resolve a Python import rather than a
#: build artefact.
GATED_LANES: tuple[str, ...] = ("go", "mojo")


#: The accelerator modules' own availability predicate. Thirty of them expose
#: it, and it is what `select_backend_order` consults, so asking it cannot
#: disagree with the dispatch under test.
AVAILABILITY_PREDICATE = "backend_available"


def _predicate(accel: ModuleType) -> Callable[[str], bool] | None:
    """Return the accelerator module's own availability check, if it has one."""
    probe = getattr(accel, AVAILABILITY_PREDICATE, None)
    return probe if callable(probe) else None


def lane_is_built(accel: ModuleType, lane: str) -> bool:
    """Return whether *lane* is executable for the model *accel* accelerates.

    Asks the accelerator module's own ``backend_available``, which is the check
    the dispatcher makes, so this cannot disagree with the code under test. A
    module that has no such predicate counts as built, because the gate has
    nothing to say about it and must not skip on a guess.
    """
    probe = _predicate(accel)
    if probe is None:
        return True
    try:
        return bool(probe(lane))
    except Exception:
        return False


def require_native_lane(accel: ModuleType, lane: str, *, subject: str) -> None:
    """Skip the current test when *lane* was never built, unless CI requires it.

    Parameters
    ----------
    accel : module
        The model's accelerator module, the one the test dispatches through.
    lane : str
        The backend name from the test's parametrisation.
    subject : str
        What the lane would have run, named in the skip reason so a reader of
        the report knows which model went unexercised.

    Raises
    ------
    RuntimeError
        When ``SC_NEUROCORE_REQUIRE_<LANE>=1`` is set and the lane is absent.
        Hosted CI sets it, so an unbuilt lane fails there rather than skipping.
    """
    if lane not in GATED_LANES:
        return
    if lane_is_built(accel, lane):
        return
    if os.environ.get(f"SC_NEUROCORE_REQUIRE_{lane.upper()}") == "1":
        raise RuntimeError(
            f"the {lane} lane is required here and is not built for {subject}; "
            f"build it or unset SC_NEUROCORE_REQUIRE_{lane.upper()}"
        )
    pytest.skip(f"the {lane} lane is not built for {subject}")


__all__ = [
    "AVAILABILITY_PREDICATE",
    "GATED_LANES",
    "lane_is_built",
    "require_native_lane",
]

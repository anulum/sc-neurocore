# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Which backend lane a model's last complete run executed

"""Record the backend lane a model's last complete run executed.

A model that offers compiled lanes records, together with the state update of
a successful run, the lane that ran and the native library files that lane
loaded. A test fixture reads the record and writes it into the test's report,
so that a readiness receipt names the lane that ran, not the one a test asked
for.

The record is kept here, keyed by the model object, and not as an attribute of
the model: code elsewhere reads every attribute a model carries (the Rust
dispatch of a population, the quiescence signature, the Studio's mutation
audit), and a lane attribute would change what they decide. A record goes when
its model goes.

Only a complete run records: it is the path that chooses among the lanes. A
single ``step()`` runs the reference lane and records nothing, so a test that
only steps a model has no lane evidence, and a validator of a lane runs the
model's complete simulation.
"""

from __future__ import annotations

import weakref
from dataclasses import dataclass
from pathlib import Path

__all__ = ["LaneRun", "lane_run", "record_lane_run"]


@dataclass(frozen=True)
class LaneRun:
    """The lane a complete run executed and the native files it loaded.

    Attributes
    ----------
    lane : str
        The backend lane that ran, for example ``python`` or ``rust``.
    libraries : tuple of Path
        The native library files the lane loaded; empty for a lane that loads
        none of its own.
    """

    lane: str
    libraries: tuple[Path, ...]


# Keyed by the identity of a live model; ``weakref.finalize`` removes the entry
# when the model is collected, before its identity can be reused.
_RUNS: dict[int, LaneRun] = {}


def record_lane_run(model: object, lane: str, libraries: tuple[Path, ...]) -> None:
    """Record the lane a model's complete run executed, replacing an earlier record.

    Parameters
    ----------
    model : object
        The model that ran. It must support weak references, so that the record
        cannot outlive it.
    lane : str
        The lane that ran.
    libraries : tuple of Path
        The native library files that lane loaded.

    Raises
    ------
    TypeError
        If the model does not support weak references.
    """
    key = id(model)
    if key not in _RUNS:
        try:
            weakref.finalize(model, _RUNS.pop, key, None)
        except TypeError as error:
            raise TypeError(
                "a model whose lane is recorded must support weak references, "
                f"got {type(model).__name__}"
            ) from error
    _RUNS[key] = LaneRun(lane, tuple(libraries))


def lane_run(model: object) -> LaneRun | None:
    """Return the lane record of a model's last complete run.

    Parameters
    ----------
    model : object
        A model.

    Returns
    -------
    LaneRun or None
        The record, or ``None`` when the model has not completed a recorded run.
    """
    return _RUNS.get(id(model))

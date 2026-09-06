# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Foreign runtime state transport contract

"""What a foreign runtime carries across the boundary, and what it may name.

A native lane returns arrays. Nothing about those arrays says which of the
model's declared state variables they are, how many samples were meant to come
back, or whether the indices in them fall inside the run. The Studio's Rust
batch lane returned its ``soma_voltage()`` trace and the payload placed it under
``states["v"]`` unconditionally.

Measured over the catalogue before this module existed: the lane runs 158 of the
185 models; of the 152 with a declared layout it carries 111 declared variables
and drops 356; and for 41 of them the model declares no variable called ``v`` at
all — ``PinskyRinzelNeuron`` declares ``v_s, v_d, h, n, s, c, q, ca`` — so the
payload named a trace after a variable that model does not have. The layout
beside it correctly reported ``recorded: []``, which made the payload and its
own custody verdict disagree.

Two further gaps at the same boundary, both measured: asking for 100 steps with
a ten-sample drive returns ten voltages while the payload is built for 100, so
the reported rate would cover a duration the trace never spanned; and the spike
indices are ``uint64`` values used directly as sample positions without a check
that they fall inside the run or arrive in order.

A packet states what a runtime transports. Coverage against the model's declared
layout decides what it may name. The validators check structure — length, shape,
dtype, index domain — before anything is built from it. What a value *means*
stays with the caller: a non-finite membrane voltage is a failed simulation, not
a malformed boundary, and is reported as one.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

#: Contract version of the packet and its coverage report.
RUNTIME_STATE_PACKET_SCHEMA_VERSION = "sc-neurocore.runtime-state-packet.v1"


class ForeignRuntimeError(ValueError):
    """Raised when a foreign runtime returns something the boundary cannot use."""


@dataclass(frozen=True, slots=True)
class RuntimeStatePacket:
    """What one foreign runtime transports across the boundary.

    Attributes
    ----------
    runtime : str
        Stable lane identifier, such as ``rust-batch``.
    exports : tuple of str
        Declared-state variable names the lane can return, in the model's own
        vocabulary. A lane that returns a value it cannot name in that
        vocabulary declares nothing for it.
    carries_initial_snapshot : bool
        Whether the lane reports the state the run started from.
    carries_parameters : bool
        Whether the lane accepts parameter overrides. A lane that does not runs
        the model's defaults whatever the caller asked for.
    """

    runtime: str
    exports: tuple[str, ...]
    carries_initial_snapshot: bool
    carries_parameters: bool

    def to_public_dict(self) -> dict[str, object]:
        """Return the packet as evidence and receipts record it."""
        return {
            "carries_initial_snapshot": self.carries_initial_snapshot,
            "carries_parameters": self.carries_parameters,
            "exports": list(self.exports),
            "runtime": self.runtime,
            "schema_version": RUNTIME_STATE_PACKET_SCHEMA_VERSION,
        }


#: The Studio's Rust batch lane. It runs the model from its own defaults and
#: returns one scalar trace, the soma voltage, plus spike indices.
RUST_BATCH_PACKET = RuntimeStatePacket(
    runtime="rust-batch",
    exports=("v",),
    carries_initial_snapshot=False,
    carries_parameters=False,
)


@dataclass(frozen=True, slots=True)
class PacketCoverage:
    """How much of one model's declared state a packet accounts for.

    Attributes
    ----------
    runtime : str
        The lane the coverage was computed for.
    carried : tuple of str
        Declared variables the lane returns, in declaration order.
    dropped : tuple of str
        Declared variables it does not return.
    unnameable : tuple of str
        Values the lane exports that this model does not declare. They cannot
        be placed in the payload under those names, because the model has no
        such state.
    """

    runtime: str
    carried: tuple[str, ...]
    dropped: tuple[str, ...]
    unnameable: tuple[str, ...]

    @property
    def complete(self) -> bool:
        """Return whether the lane carries every declared variable."""
        return not self.dropped and not self.unnameable

    @property
    def names_nothing(self) -> bool:
        """Return whether the lane can name none of this model's state."""
        return not self.carried

    def to_public_dict(self) -> dict[str, object]:
        """Return the coverage as evidence and receipts record it."""
        return {
            "carried": list(self.carried),
            "complete": self.complete,
            "dropped": list(self.dropped),
            "runtime": self.runtime,
            "schema_version": RUNTIME_STATE_PACKET_SCHEMA_VERSION,
            "unnameable": list(self.unnameable),
        }


def packet_coverage(packet: RuntimeStatePacket, declared_names: Sequence[str]) -> PacketCoverage:
    """Return what a packet accounts for against one model's declared state.

    Parameters
    ----------
    packet : RuntimeStatePacket
        The lane's transport contract.
    declared_names : sequence of str
        The model's declared state variables, in declaration order.

    Returns
    -------
    PacketCoverage
        Which declared variables the lane carries, which it drops, and which of
        its exports this model cannot name.
    """
    declared = tuple(declared_names)
    exported = set(packet.exports)
    carried = tuple(name for name in declared if name in exported)
    dropped = tuple(name for name in declared if name not in exported)
    unnameable = tuple(name for name in packet.exports if name not in set(declared))
    return PacketCoverage(
        runtime=packet.runtime, carried=carried, dropped=dropped, unnameable=unnameable
    )


def validate_scalar_trace(
    values: object, *, runtime: str, name: str, n_steps: int
) -> np.ndarray[Any, np.dtype[np.float64]]:
    """Return one scalar trace from a foreign runtime, or refuse it.

    Structure only: the trace must be one-dimensional real numbers of exactly
    the requested length. Whether the values are finite is the caller's
    question, because a diverged membrane voltage is a failed simulation rather
    than a malformed boundary.

    Parameters
    ----------
    values : object
        Whatever the lane returned for this trace.
    runtime : str
        Lane identifier, named in the refusal.
    name : str
        Variable name, named in the refusal.
    n_steps : int
        Number of steps the run asked for.

    Returns
    -------
    numpy.ndarray
        The trace as float64.

    Raises
    ------
    ForeignRuntimeError
        If the trace is not one-dimensional real numbers of length ``n_steps``.
    """
    try:
        trace = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ForeignRuntimeError(
            f"{runtime} returned a {name!r} trace that is not numeric."
        ) from exc
    if trace.ndim != 1:
        raise ForeignRuntimeError(
            f"{runtime} returned a {name!r} trace of shape {trace.shape}, not a single series."
        )
    if trace.shape[0] != n_steps:
        raise ForeignRuntimeError(
            f"{runtime} returned {trace.shape[0]} samples of {name!r} for a {n_steps}-step run."
        )
    return trace


def validate_spike_indices(values: object, *, runtime: str, n_steps: int) -> list[int]:
    """Return spike sample positions from a foreign runtime, or refuse them.

    Parameters
    ----------
    values : object
        Whatever the lane returned for its spike indices.
    runtime : str
        Lane identifier, named in the refusal.
    n_steps : int
        Number of steps the run asked for; every index must fall inside it.

    Returns
    -------
    list of int
        The indices, in the order the lane produced them.

    Raises
    ------
    ForeignRuntimeError
        If the indices are not a one-dimensional integer series inside the run
        and in increasing order. Order matters: the statistics take differences
        between consecutive entries, and an unordered series would report a
        negative interval as a real measurement.
    """
    try:
        indices = np.asarray(values)
    except (TypeError, ValueError) as exc:
        raise ForeignRuntimeError(
            f"{runtime} returned spike indices that are not an array."
        ) from exc
    if indices.ndim != 1:
        raise ForeignRuntimeError(
            f"{runtime} returned spike indices of shape {indices.shape}, not a single series."
        )
    if indices.size and indices.dtype.kind not in "iu":
        raise ForeignRuntimeError(
            f"{runtime} returned spike indices of dtype {indices.dtype}, which are not positions."
        )
    positions = [int(index) for index in indices.tolist()]
    for position in positions:
        if not 0 <= position < n_steps:
            raise ForeignRuntimeError(
                f"{runtime} returned the spike index {position}, outside a {n_steps}-step run."
            )
    if any(later <= earlier for earlier, later in zip(positions, positions[1:])):
        raise ForeignRuntimeError(f"{runtime} returned spike indices that do not increase.")
    return positions


__all__ = [
    "RUNTIME_STATE_PACKET_SCHEMA_VERSION",
    "RUST_BATCH_PACKET",
    "ForeignRuntimeError",
    "PacketCoverage",
    "RuntimeStatePacket",
    "packet_coverage",
    "validate_scalar_trace",
    "validate_spike_indices",
]

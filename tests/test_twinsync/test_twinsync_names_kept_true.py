# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Twin-sync names that now say what the code does

"""Hold the twin-sync names that promised more than the code to what they now do.

The checkpoint interval counted events under a nanosecond name; the checkpoint checksum covered four fields and let
states of another shape, or None and an empty array, hash equal; the delta's compression ratio was a constant; the
session's ``max_divergence`` changed nothing; and ``digital_events_out`` was never counted. These tests hold the
event-count name with its deprecated alias, a checksum over every stored field, a real compression ratio, a session
tolerance taken from ``max_divergence``, and the removal of the uncounted counter.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import pytest

from sc_neurocore.digital_twin.twinsync import (
    Checkpoint,
    DeltaCheckpoint,
    DivergenceMetric,
    SessionSnapshot,
    TimeWarpEngine,
    TwinEvent,
    TwinSession,
)


def test_the_checkpoint_interval_counts_events_under_its_own_name() -> None:
    """The second positional argument keeps its meaning; the old keyword warns; both together are refused."""
    assert TimeWarpEngine(2, 3).checkpoint_interval_events == 3
    assert TimeWarpEngine(2).checkpoint_interval_events == 1000
    with pytest.warns(DeprecationWarning, match="checkpoint_interval_events"):
        engine = TimeWarpEngine(2, checkpoint_interval_ns=5)
    assert engine.checkpoint_interval_events == 5
    with pytest.warns(DeprecationWarning):
        assert engine.checkpoint_interval_ns == 5
    with pytest.raises(ValueError, match="deprecated name"):
        TimeWarpEngine(2, 3, checkpoint_interval_ns=5)


def _checkpoint(**fields: Any) -> Checkpoint:
    """Return a fixed checkpoint with ``fields`` replaced."""
    return dataclasses.replace(Checkpoint(checkpoint_id=1, virtual_time_ns=10, node_id=0), **fields)


@pytest.mark.parametrize(
    "fields",
    [
        {"node_id": 1},
        {"lamport_time": 7},
        {"identity_deep": 0.5},
        {"synapse_state": np.ones(3)},
        {"vector_clock": np.array([1, 0])},
    ],
)
def test_the_checksum_covers_every_stored_field(fields: dict[str, Any]) -> None:
    """Changing the node, Lamport time, identity, synapse state or vector clock changes the checksum."""
    assert _checkpoint(**fields).compute_checksum() != _checkpoint().compute_checksum()


def test_the_checksum_tells_shapes_and_none_apart() -> None:
    """The same bytes in another shape, and None against an empty array, hash differently."""
    flat = np.arange(6.0)
    tall = _checkpoint(neuron_state=flat.reshape(2, 3)).compute_checksum()
    wide = _checkpoint(neuron_state=flat.reshape(3, 2)).compute_checksum()
    assert tall != wide
    assert (
        _checkpoint().compute_checksum() != _checkpoint(neuron_state=np.empty(0)).compute_checksum()
    )


def test_the_compression_ratio_is_full_bytes_over_delta_bytes() -> None:
    """One change in 100 float64 elements stores 16 bytes for 800; changing every element costs more than the state."""
    base = np.zeros(100)
    one = base.copy()
    one[7] = 1.0
    assert DeltaCheckpoint.compute_delta(base, one, 0, 1, 0, 0).compression_ratio == pytest.approx(
        50.0
    )
    assert DeltaCheckpoint.compute_delta(
        base, np.ones(100), 0, 1, 0, 0
    ).compression_ratio == pytest.approx(0.5)
    assert DeltaCheckpoint.compute_delta(base, base.copy(), 0, 1, 0, 0).compression_ratio == 0.0


def test_a_delta_cannot_be_built_without_its_state_shape() -> None:
    """The ratio needs the shape: it is required and keyword only, so an old positional call fails loudly."""
    required: tuple[Any, ...] = (0, 1, 0, 0, np.zeros(0, dtype=np.int64), np.zeros(0))
    no_keywords: dict[str, Any] = {}
    with pytest.raises(TypeError):
        DeltaCheckpoint(*required, **no_keywords)
    old_positional: tuple[Any, ...] = (*required, 0, 0)
    with pytest.raises(TypeError):
        DeltaCheckpoint(*old_positional, **no_keywords)
    delta = DeltaCheckpoint(*old_positional, state_shape=(0,))
    assert (delta.lfsr_delta, delta.size_bytes, delta.state_shape) == (0, 0, (0,))


def test_the_session_tolerance_is_its_max_divergence() -> None:
    """``within_tolerance`` compares with the session's ``max_divergence``; a bare metric keeps 1.0."""
    assert DivergenceMetric(spike_rate_divergence=0.5).within_tolerance is True
    strict = TwinSession(1, max_divergence=0.25)
    metric = strict.update_divergence(
        physical_rate_hz=10.0, digital_rate_hz=7.0, physical_identity=0.0
    )
    assert metric.tolerance == 0.25
    assert metric.within_tolerance is False
    loose = TwinSession(1, max_divergence=2.0)
    assert loose.update_divergence(10.0, 7.0, 0.0).within_tolerance is True


def test_the_uncounted_outbound_counter_is_gone() -> None:
    """Nothing leaves the twin: the counter is out of the status and the snapshot, and its attribute warns and is 0."""
    session = TwinSession(1)
    session.engine.inject_event(TwinEvent(10, target_node=0))
    session.advance()
    assert "digital_events" not in session.status()
    assert "digital_events_out" not in SessionSnapshot.from_session(session).to_dict()
    with pytest.warns(DeprecationWarning, match="never counted"):
        assert session.digital_events_out == 0


def test_the_global_virtual_time_can_fall() -> None:
    """A straggler queued below every local time lowers the global virtual time."""
    engine = TimeWarpEngine(1)
    engine.inject_event(TwinEvent(200, target_node=0))
    engine.process_next()
    assert engine.compute_gvt() == 200
    engine.inject_event(TwinEvent(100, target_node=0))
    assert engine.compute_gvt() == 100

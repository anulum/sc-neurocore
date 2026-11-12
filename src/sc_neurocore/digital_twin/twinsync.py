# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Brain-Scale Digital-Twin Synchronization Primitives

"""SC-native time-warp synchronization for BCI digital twins.

Provides causal ordering, checkpoint/resume, and time-warp rollback
across distributed MPI nodes for real-time synchronization between
a physical BCI subject and its billion-neuron SC digital twin.

Key primitives:
- **Lamport Clock**: Logical timestamps for causal ordering
- **Vector Clock**: Full causal dependency tracking across N nodes
- **Time-Warp Engine**: Optimistic execution with anti-message rollback
- **Checkpoint Manager**: Deterministic state snapshots with LFSR replay
- **Twin Session**: Orchestrates physical ↔ digital synchronization

Architecture:

    Physical BCI Subject
           │ MEA / EEG / fNIRS
    ┌──────▼──────┐
    │ SensorBridge │──► AER spike stream
    └──────┬──────┘
           │ causal-ordered events
    ┌──────▼──────────────────────────┐
    │     TwinSync Time-Warp Engine    │
    │  ┌────────┐ ┌────────┐ ┌──────┐│
    │  │ Node 0 │ │ Node 1 │ │Node N││
    │  │ 100M   │ │ 100M   │ │100M  ││
    │  │neurons │ │neurons │ │neuro ││
    │  └───┬────┘ └───┬────┘ └──┬───┘│
    │      └──────────┴─────────┘    │
    │         vector clocks          │
    └──────┬─────────────────────────┘
           │ synchronized output
    ┌──────▼──────┐
    │ OutputBridge │──► closed-loop stimulation
    └─────────────┘

Compatible with:
- ``mpi_partitioner.py`` / ``hierarchical_partitioner.py`` — node topology
- ``identity substrate`` (ArcaneNeuron.v_deep) — preserved across rollback
- ``bioware/`` — MEA/AER physical interface
- ``sc_scope/`` — live monitoring of twin divergence
"""

from __future__ import annotations

import hashlib
import heapq
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ── Logical Clocks ───────────────────────────────────────────────────


class LamportClock:
    """Lamport logical clock for causal ordering.

    Attributes
    ----------
    time : int
        The current logical time; starts at 0.
    """

    def __init__(self) -> None:
        self.time: int = 0

    def tick(self) -> int:
        """Advance the clock for a local event.

        Returns
        -------
        int
            The new logical time.
        """
        self.time += 1
        return self.time

    def send(self) -> int:
        """Advance the clock and return the timestamp a sent message carries.

        Returns
        -------
        int
            The new logical time, to attach to the outgoing message.
        """
        self.time += 1
        return self.time

    def receive(self, remote_time: int) -> int:
        """Merge a received timestamp: one past the larger of both clocks.

        Parameters
        ----------
        remote_time : int
            The logical time the received message carries.

        Returns
        -------
        int
            The new logical time.
        """
        self.time = max(self.time, remote_time) + 1
        return self.time


class VectorClock:
    """Vector clock for full causal dependency tracking.

    Parameters
    ----------
    node_id : int
        Index of the owning node in the vector.
    num_nodes : int
        Number of nodes, the length of the vector.

    Attributes
    ----------
    clock : numpy.ndarray
        One int64 counter per node, all 0 at the start.
    """

    def __init__(self, node_id: int, num_nodes: int):
        self.node_id = node_id
        self.num_nodes = num_nodes
        self.clock = np.zeros(num_nodes, dtype=np.int64)

    def tick(self) -> np.ndarray[Any, Any]:
        """Advance the owning node's counter for a local event.

        Returns
        -------
        numpy.ndarray
            A copy of the clock after the increment.
        """
        self.clock[self.node_id] += 1
        return self.clock.copy()

    def send(self) -> np.ndarray[Any, Any]:
        """Advance the owning node's counter and return the vector a message carries.

        Returns
        -------
        numpy.ndarray
            A copy of the clock after the increment.
        """
        self.clock[self.node_id] += 1
        return self.clock.copy()

    def receive(self, remote_clock: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Merge a received vector element by element, then advance the own counter.

        Parameters
        ----------
        remote_clock : numpy.ndarray
            The vector the received message carries, one counter per node.

        Returns
        -------
        numpy.ndarray
            A copy of the clock after the merge and the increment.
        """
        self.clock = np.maximum(self.clock, remote_clock)
        self.clock[self.node_id] += 1
        return self.clock.copy()

    def happened_before(self, other: np.ndarray[Any, Any]) -> bool:
        """Tell whether this clock happened before another vector.

        Parameters
        ----------
        other : numpy.ndarray
            The vector to compare with.

        Returns
        -------
        bool
            True when every counter is at most the other's and at least one is
            strictly smaller.
        """
        return bool(np.all(self.clock <= other) and np.any(self.clock < other))

    def concurrent_with(self, other: np.ndarray[Any, Any]) -> bool:
        """Tell whether this clock and another vector are causally unordered.

        Parameters
        ----------
        other : numpy.ndarray
            The vector to compare with.

        Returns
        -------
        bool
            True when neither happened before the other.
        """
        return not self.happened_before(other) and not bool(
            np.all(other <= self.clock) and np.any(other < self.clock)
        )


# ── Events ───────────────────────────────────────────────────────────


class EventType(Enum):
    """Kinds of event the time-warp engine queues."""

    SPIKE = "spike"
    SENSOR_INPUT = "sensor_input"
    CHECKPOINT = "checkpoint"
    ANTI_MESSAGE = "anti_message"
    ROLLBACK = "rollback"
    SYNC_BARRIER = "sync_barrier"


@dataclass(order=True)
class TwinEvent:
    """One event in the time-warp simulation.

    Events order by virtual time, then by priority; the other fields do not
    take part in the ordering.

    Attributes
    ----------
    virtual_time_ns : int
        Virtual time of the event in nanoseconds.
    priority : int
        Tie-breaker between events of the same virtual time.
    event_type : EventType
        What the event is.
    source_node, target_node : int
        Sending and receiving node; -1 as source marks the physical world or a
        global barrier.
    payload : dict
        Event data, for example the neuron id of a sensor event.
    lamport_ts : int
        Lamport timestamp the event carries.
    vector_ts : numpy.ndarray or None
        Vector timestamp the event carries, if any.
    cancelled : bool
        A cancelled event is dequeued without being processed.
    """

    virtual_time_ns: int
    priority: int = field(compare=True, default=0)
    event_type: EventType = field(compare=False, default=EventType.SPIKE)
    source_node: int = field(compare=False, default=0)
    target_node: int = field(compare=False, default=0)
    payload: Dict[str, Any] = field(compare=False, default_factory=dict)
    lamport_ts: int = field(compare=False, default=0)
    vector_ts: Optional[np.ndarray[Any, Any]] = field(compare=False, default=None)
    cancelled: bool = field(compare=False, default=False)


# ── Checkpoint Manager ──────────────────────────────────────────────


@dataclass
class Checkpoint:
    """Deterministic state snapshot for rollback.

    Attributes
    ----------
    checkpoint_id : int
        Sequence number given by the checkpoint manager.
    virtual_time_ns : int
        Virtual time of the snapshot in nanoseconds.
    node_id : int
        Node whose state was saved.
    neuron_state, synapse_state : numpy.ndarray or None
        Copies of the saved state arrays.
    lfsr_state : int
        Saved random-number-generator state.
    identity_deep : float
        The node's deep identity value at the time of the snapshot.
    lamport_time : int
        The node's Lamport time.
    vector_clock : numpy.ndarray or None
        A copy of the node's vector clock.
    checksum : str
        Set by :meth:`compute_checksum`.
    """

    checkpoint_id: int
    virtual_time_ns: int
    node_id: int
    neuron_state: Optional[np.ndarray[Any, Any]] = None
    synapse_state: Optional[np.ndarray[Any, Any]] = None
    lfsr_state: int = 0
    identity_deep: float = 0.0
    lamport_time: int = 0
    vector_clock: Optional[np.ndarray[Any, Any]] = None
    checksum: str = ""

    def compute_checksum(self) -> str:
        """Hash the snapshot's identifying fields and neuron state, and store the result.

        The SHA-256 covers the checkpoint id (4 bytes), the virtual time (8 bytes),
        the LFSR state (4 bytes, all little-endian) and the raw bytes of the neuron
        state. The synapse state, identity, Lamport time and vector clock are not
        part of it.

        Returns
        -------
        str
            The first 16 hexadecimal characters of the digest, also stored in
            :attr:`checksum`.

        Raises
        ------
        OverflowError
            If a field is negative or does not fit its byte width.
        """
        h = hashlib.sha256()
        h.update(self.checkpoint_id.to_bytes(4, "little"))
        h.update(self.virtual_time_ns.to_bytes(8, "little"))
        h.update(self.lfsr_state.to_bytes(4, "little"))
        if self.neuron_state is not None:
            h.update(self.neuron_state.tobytes())
        self.checksum = h.hexdigest()[:16]
        return self.checksum


class CheckpointManager:
    """Manages state snapshots for time-warp rollback.

    Preserves the identity substrate (ArcaneNeuron.v_deep) across
    rollback: deep compartment is NEVER rolled back.

    Parameters
    ----------
    max_checkpoints : int
        Checkpoints kept per node; older ones are dropped on save.
    """

    def __init__(self, max_checkpoints: int = 100):
        self.max_checkpoints = max_checkpoints
        self.checkpoints: Dict[int, List[Checkpoint]] = {}  # node_id → sorted list
        self._next_id: int = 0

    def save(
        self,
        node_id: int,
        virtual_time_ns: int,
        neuron_state: Optional[np.ndarray[Any, Any]] = None,
        synapse_state: Optional[np.ndarray[Any, Any]] = None,
        lfsr_state: int = 0,
        identity_deep: float = 0.0,
        lamport_time: int = 0,
        vector_clock: Optional[np.ndarray[Any, Any]] = None,
    ) -> Checkpoint:
        """Save a snapshot of one node, keeping at most ``max_checkpoints`` per node.

        The arrays are copied. The checkpoint gets the next sequence number and its
        checksum.

        Parameters
        ----------
        node_id : int
            Node whose state is saved.
        virtual_time_ns : int
            Virtual time of the snapshot.
        neuron_state, synapse_state : numpy.ndarray or None
            State arrays to copy.
        lfsr_state : int
            Random-number-generator state.
        identity_deep : float
            The node's deep identity value.
        lamport_time : int
            The node's Lamport time.
        vector_clock : numpy.ndarray or None
            The node's vector clock, copied.

        Returns
        -------
        Checkpoint
            The saved checkpoint.
        """
        cp = Checkpoint(
            checkpoint_id=self._next_id,
            virtual_time_ns=virtual_time_ns,
            node_id=node_id,
            neuron_state=neuron_state.copy() if neuron_state is not None else None,
            synapse_state=synapse_state.copy() if synapse_state is not None else None,
            lfsr_state=lfsr_state,
            identity_deep=identity_deep,
            lamport_time=lamport_time,
            vector_clock=vector_clock.copy() if vector_clock is not None else None,
        )
        cp.compute_checksum()
        self._next_id += 1

        if node_id not in self.checkpoints:
            self.checkpoints[node_id] = []
        self.checkpoints[node_id].append(cp)

        # Garbage collection: keep only latest N
        if len(self.checkpoints[node_id]) > self.max_checkpoints:
            self.checkpoints[node_id] = self.checkpoints[node_id][-self.max_checkpoints :]

        return cp

    def find_rollback_target(self, node_id: int, target_time_ns: int) -> Optional[Checkpoint]:
        """Return the last checkpoint saved for a node at or before a virtual time.

        Parameters
        ----------
        node_id : int
            Node to look up.
        target_time_ns : int
            Latest acceptable virtual time.

        Returns
        -------
        Checkpoint or None
            The last such checkpoint in save order, or None when there is none.
        """
        cps = self.checkpoints.get(node_id, [])
        best = None
        for cp in cps:
            if cp.virtual_time_ns <= target_time_ns:
                best = cp
        return best

    def discard_after(self, node_id: int, time_ns: int) -> int:
        """Drop a node's checkpoints later than a virtual time, after a rollback.

        Parameters
        ----------
        node_id : int
            Node whose checkpoints are pruned.
        time_ns : int
            Checkpoints with a later virtual time are dropped.

        Returns
        -------
        int
            How many checkpoints were dropped.
        """
        cps = self.checkpoints.get(node_id, [])
        before = len(cps)
        self.checkpoints[node_id] = [cp for cp in cps if cp.virtual_time_ns <= time_ns]
        return before - len(self.checkpoints.get(node_id, []))

    @property
    def total_checkpoints(self) -> int:
        """Number of checkpoints held for all nodes together."""
        return sum(len(v) for v in self.checkpoints.values())


# ── Time-Warp Engine ────────────────────────────────────────────────


@dataclass
class NodeState:
    """Per-node state in the time-warp simulation.

    Attributes
    ----------
    node_id : int
        Index of the node.
    local_virtual_time_ns : int
        The node's local virtual time.
    lamport : LamportClock
        The node's Lamport clock.
    vector_clock : VectorClock or None
        The node's vector clock.
    processed_events : int
        Events the node has processed.
    rollback_count : int
        Rollbacks the node has made.
    identity_deep : float
        Deep identity value; a rollback never changes it.
    """

    node_id: int
    local_virtual_time_ns: int = 0
    lamport: LamportClock = field(default_factory=LamportClock)
    vector_clock: Optional[VectorClock] = None
    processed_events: int = 0
    rollback_count: int = 0
    identity_deep: float = 0.0  # Never rolled back


class TimeWarpEngine:
    """Optimistic parallel simulation with anti-message rollback.

    Implements the Jefferson Time Warp protocol adapted for
    SC neuromorphic simulation:

    1. Each node advances optimistically at its own rate
    2. Straggler events trigger rollback + anti-messages
    3. Global Virtual Time (GVT) advances monotonically
    4. Fossil collection prunes checkpoints below GVT
    5. Identity (v_deep) is NEVER rolled back — it is the self

    Parameters
    ----------
    num_nodes : int
        Number of nodes, each with a Lamport and a vector clock.
    checkpoint_interval_ns : int
        Despite its name, a count of events: a node saves a checkpoint each time
        its number of processed events is a multiple of this value (values below
        1 count as 1).
    """

    def __init__(self, num_nodes: int, checkpoint_interval_ns: int = 1000):
        self.num_nodes = num_nodes
        self.checkpoint_interval_ns = checkpoint_interval_ns
        self.nodes: Dict[int, NodeState] = {}
        for i in range(num_nodes):
            ns = NodeState(node_id=i)
            ns.vector_clock = VectorClock(i, num_nodes)
            self.nodes[i] = ns
        self.event_queue: List[TwinEvent] = []
        self.processed: List[TwinEvent] = []
        self.anti_messages: List[TwinEvent] = []
        self.checkpoint_mgr = CheckpointManager()
        self.gvt_ns: int = 0
        self.total_rollbacks: int = 0

    def inject_event(self, event: TwinEvent) -> None:
        """Queue an event in virtual-time order.

        Parameters
        ----------
        event : TwinEvent
            The event to queue.
        """
        heapq.heappush(self.event_queue, event)

    def process_next(self) -> Optional[TwinEvent]:
        """Take the earliest event from the queue and apply it to its target node.

        A cancelled event, or one whose target node does not exist, is returned
        without being applied. An event earlier than the target's local virtual
        time is a straggler: the node is rolled back first. Applying an event sets
        the node's local time to the event's, merges its Lamport and vector
        timestamps and counts it; every ``checkpoint_interval_ns``-th event of a
        node saves a checkpoint.

        Returns
        -------
        TwinEvent or None
            The dequeued event, or None when the queue is empty.
        """
        if not self.event_queue:
            return None

        event = heapq.heappop(self.event_queue)
        if event.cancelled:
            return event

        target = self.nodes.get(event.target_node)
        if target is None:
            return event

        # Check for straggler (causality violation)
        if event.virtual_time_ns < target.local_virtual_time_ns:
            self._rollback(target, event.virtual_time_ns)

        # Process event
        target.local_virtual_time_ns = event.virtual_time_ns
        target.lamport.receive(event.lamport_ts)
        if target.vector_clock is not None and event.vector_ts is not None:
            target.vector_clock.receive(event.vector_ts)
        target.processed_events += 1

        # Periodic checkpoint
        if target.processed_events % max(1, self.checkpoint_interval_ns) == 0:
            self.checkpoint_mgr.save(
                target.node_id,
                target.local_virtual_time_ns,
                lfsr_state=target.processed_events,
                identity_deep=target.identity_deep,
                lamport_time=target.lamport.time,
                vector_clock=target.vector_clock.clock if target.vector_clock else None,
            )

        self.processed.append(event)
        return event

    def _rollback(self, node: NodeState, target_time_ns: int) -> None:
        """Roll back a node to a checkpoint at or before target_time.

        Identity (v_deep) is preserved — never rolled back.
        """
        saved_identity = node.identity_deep

        cp = self.checkpoint_mgr.find_rollback_target(node.node_id, target_time_ns)
        if cp is not None:
            node.local_virtual_time_ns = cp.virtual_time_ns
            node.lamport.time = cp.lamport_time
            if node.vector_clock is not None and cp.vector_clock is not None:
                node.vector_clock.clock = cp.vector_clock.copy()
            self.checkpoint_mgr.discard_after(node.node_id, cp.virtual_time_ns)
        else:
            node.local_virtual_time_ns = target_time_ns

        # Restore identity
        node.identity_deep = saved_identity
        node.rollback_count += 1
        self.total_rollbacks += 1

        # Generate anti-messages for events processed after rollback point
        anti = [
            TwinEvent(
                virtual_time_ns=e.virtual_time_ns,
                event_type=EventType.ANTI_MESSAGE,
                source_node=node.node_id,
                target_node=e.target_node,
                lamport_ts=node.lamport.send(),
            )
            for e in self.processed
            if e.source_node == node.node_id and e.virtual_time_ns > target_time_ns
        ]
        self.anti_messages.extend(anti)
        for a in anti:
            heapq.heappush(self.event_queue, a)

    def compute_gvt(self) -> int:
        """Compute the global virtual time and store it.

        Returns
        -------
        int
            The smallest of all nodes' local virtual times and the virtual times of
            the queued events that are not cancelled; 0 when there are none.
        """
        lvts = [n.local_virtual_time_ns for n in self.nodes.values()]
        in_transit = [e.virtual_time_ns for e in self.event_queue if not e.cancelled]
        all_times = lvts + in_transit
        self.gvt_ns = min(all_times) if all_times else 0
        return self.gvt_ns

    def fossil_collect(self) -> int:
        """Drop every checkpoint older than the global virtual time.

        Returns
        -------
        int
            How many checkpoints were dropped.
        """
        gvt = self.compute_gvt()
        removed = 0
        for nid in list(self.checkpoint_mgr.checkpoints.keys()):
            cps = self.checkpoint_mgr.checkpoints[nid]
            before = len(cps)
            self.checkpoint_mgr.checkpoints[nid] = [cp for cp in cps if cp.virtual_time_ns >= gvt]
            removed += before - len(self.checkpoint_mgr.checkpoints[nid])
        return removed

    def status(self) -> Dict[str, Any]:
        """Return the engine's counters and each node's local virtual time.

        Returns
        -------
        dict
            Number of nodes, global virtual time, rollbacks, queued and processed
            events, checkpoints held and local virtual time per node.
        """
        return {
            "num_nodes": self.num_nodes,
            "gvt_ns": self.gvt_ns,
            "total_rollbacks": self.total_rollbacks,
            "pending_events": len(self.event_queue),
            "processed_events": len(self.processed),
            "checkpoints": self.checkpoint_mgr.total_checkpoints,
            "node_lvts": {nid: n.local_virtual_time_ns for nid, n in self.nodes.items()},
        }

    def inject_sync_barrier(self, virtual_time_ns: int) -> None:
        """Queue one synchronisation barrier event for every node at a virtual time.

        Parameters
        ----------
        virtual_time_ns : int
            Virtual time of the barrier.
        """
        for nid in self.nodes:
            event = TwinEvent(
                virtual_time_ns=virtual_time_ns,
                event_type=EventType.SYNC_BARRIER,
                source_node=-1,
                target_node=nid,
                lamport_ts=0,
            )
            heapq.heappush(self.event_queue, event)

    def verify_causal_order(self) -> List[Tuple[int, int]]:
        """Find neighbouring processed events of one node that ran backwards in time.

        Only consecutive entries of the processed list are compared.

        Returns
        -------
        list of tuple of int
            Index pairs ``(i, i + 1)`` where both events target the same node and
            the first has the later virtual time.
        """
        violations = []
        for i in range(len(self.processed) - 1):
            a = self.processed[i]
            b = self.processed[i + 1]
            if a.target_node == b.target_node and a.virtual_time_ns > b.virtual_time_ns:
                violations.append((i, i + 1))
        return violations

    def detect_starvation(self, threshold_ns: int = 10000) -> List[int]:
        """List the nodes whose local time lies more than a threshold below the global time.

        Parameters
        ----------
        threshold_ns : int
            Allowed lag in nanoseconds.

        Returns
        -------
        list of int
            Ids of the lagging nodes.
        """
        gvt = self.compute_gvt()
        return [
            nid for nid, n in self.nodes.items() if gvt - n.local_virtual_time_ns > threshold_ns
        ]

    def node_throughput(self) -> Dict[int, int]:
        """Return the number of events each node has processed.

        Returns
        -------
        dict of int to int
            Processed events by node id.
        """
        return {nid: n.processed_events for nid, n in self.nodes.items()}


# ── Twin Session ────────────────────────────────────────────────────


class SyncMode(Enum):
    """Synchronisation strategies a twin session can be labelled with."""

    LOCKSTEP = "lockstep"
    OPTIMISTIC = "optimistic"
    CONSERVATIVE = "conservative"


@dataclass
class DivergenceMetric:
    """Measures divergence between physical and digital twin.

    Attributes
    ----------
    spike_rate_divergence : float
        Relative difference of the physical and digital spike rates.
    timing_offset_ns : int
        Session time minus the engine's global virtual time, in nanoseconds.
    identity_drift : float
        Absolute difference of the physical and digital identity values.
    causal_violations : int
        The engine's rollback count.
    """

    spike_rate_divergence: float = 0.0
    timing_offset_ns: int = 0
    identity_drift: float = 0.0
    causal_violations: int = 0

    @property
    def total_divergence(self) -> float:
        """Heuristic sum of the four parts, mixing their units.

        The rate divergence, the timing offset in milliseconds, the identity drift
        and 0.1 per causal violation are added as plain numbers.
        """
        return (
            self.spike_rate_divergence
            + abs(self.timing_offset_ns) / 1e6
            + self.identity_drift
            + self.causal_violations * 0.1
        )

    @property
    def within_tolerance(self) -> bool:
        """Whether the total divergence is below 1.0."""
        return self.total_divergence < 1.0


class TwinSession:
    """Orchestrates physical ↔ digital twin synchronization.

    Manages bidirectional data flow:
    - Physical → Digital: sensor events (MEA spikes, EEG)
    - Digital → Physical: stimulation commands (opto, TMS)

    Parameters
    ----------
    num_nodes : int
        Number of nodes of the session's time-warp engine.
    mode : SyncMode
        The strategy the session reports; the engine runs optimistically in every
        mode.
    max_divergence : float
        Stored for the caller; the session does not act on it.
    """

    def __init__(
        self,
        num_nodes: int,
        mode: SyncMode = SyncMode.OPTIMISTIC,
        max_divergence: float = 1.0,
    ):
        self.num_nodes = num_nodes
        self.mode = mode
        self.max_divergence = max_divergence
        self.engine = TimeWarpEngine(num_nodes)
        self.divergence = DivergenceMetric()
        self.physical_events_in: int = 0
        self.digital_events_out: int = 0
        self.session_time_ns: int = 0
        self.running: bool = False

    def start(self) -> None:
        """Mark the session as running; nothing else changes."""
        self.running = True

    def stop(self) -> None:
        """Mark the session as stopped; nothing else changes."""
        self.running = False

    def inject_physical_event(
        self, spike_time_ns: int, neuron_id: int, target_node: int = 0
    ) -> None:
        """Queue a sensor event from the physical subject for the digital twin.

        Parameters
        ----------
        spike_time_ns : int
            Virtual time of the event.
        neuron_id : int
            Physical neuron or channel that fired; carried in the payload.
        target_node : int
            Node of the twin that receives the event.
        """
        event = TwinEvent(
            virtual_time_ns=spike_time_ns,
            event_type=EventType.SENSOR_INPUT,
            source_node=-1,  # physical world
            target_node=target_node,
            payload={"neuron_id": neuron_id},
            lamport_ts=0,
        )
        self.engine.inject_event(event)
        self.physical_events_in += 1

    def advance(self, steps: int = 1) -> int:
        """Process up to a number of queued events.

        Parameters
        ----------
        steps : int
            Most events to process; the loop stops early on an empty queue.

        Returns
        -------
        int
            Events dequeued, cancelled ones included. The session time becomes the
            latest virtual time seen.
        """
        processed = 0
        for _ in range(steps):
            ev = self.engine.process_next()
            if ev is None:
                break
            processed += 1
            self.session_time_ns = max(self.session_time_ns, ev.virtual_time_ns)
        return processed

    def update_divergence(
        self,
        physical_rate_hz: float,
        digital_rate_hz: float,
        physical_identity: float,
    ) -> DivergenceMetric:
        """Recompute the divergence between the physical subject and the twin.

        Parameters
        ----------
        physical_rate_hz : float
            Spike rate measured on the subject.
        digital_rate_hz : float
            Spike rate of the twin.
        physical_identity : float
            Identity value measured on the subject; compared with the first node's
            deep identity value.

        Returns
        -------
        DivergenceMetric
            The new metric, also stored on the session. The rate divergence is
            relative to the physical rate, at least 1 Hz.
        """
        digital_identity = 0.0
        if self.engine.nodes:
            digital_identity = list(self.engine.nodes.values())[0].identity_deep

        self.divergence = DivergenceMetric(
            spike_rate_divergence=abs(physical_rate_hz - digital_rate_hz)
            / max(physical_rate_hz, 1.0),
            timing_offset_ns=self.session_time_ns - self.engine.gvt_ns,
            identity_drift=abs(physical_identity - digital_identity),
            causal_violations=self.engine.total_rollbacks,
        )
        return self.divergence

    def status(self) -> Dict[str, Any]:
        """Return the session's state, its divergence and the engine's status.

        Returns
        -------
        dict
            Running flag, mode, session time, event counters, total divergence,
            whether it is within tolerance and :meth:`TimeWarpEngine.status`.
        """
        return {
            "running": self.running,
            "mode": self.mode.value,
            "session_time_ns": self.session_time_ns,
            "physical_events": self.physical_events_in,
            "digital_events": self.digital_events_out,
            "divergence": self.divergence.total_divergence,
            "within_tolerance": self.divergence.within_tolerance,
            "engine": self.engine.status(),
        }


# ── Null-Message Lookahead (Gap 1) ──────────────────────────────────


@dataclass
class LookaheadConfig:
    """Null-message lookahead for conservative synchronization.

    Each node declares a minimum time advance (lookahead) it guarantees
    before generating output events. Peers can safely advance by at
    least this amount without rollback risk.

    Attributes
    ----------
    node_id : int
        The declaring node.
    lookahead_ns : int
        The guaranteed advance in nanoseconds.
    last_null_message_ns : int
        Virtual time of the node's last null message.
    """

    node_id: int
    lookahead_ns: int = 1000
    last_null_message_ns: int = 0

    def can_advance_to(self, target_ns: int) -> bool:
        """Tell whether a virtual time lies within the declared lookahead.

        Parameters
        ----------
        target_ns : int
            The virtual time a peer wants to reach.

        Returns
        -------
        bool
            True when it is at most the last null message plus the lookahead.
        """
        return target_ns <= self.last_null_message_ns + self.lookahead_ns

    def send_null_message(self, current_ns: int) -> int:
        """Record a null message at a virtual time.

        Parameters
        ----------
        current_ns : int
            The node's current virtual time.

        Returns
        -------
        int
            The time up to which peers may now advance.
        """
        self.last_null_message_ns = current_ns
        return current_ns + self.lookahead_ns


class NullMessageOptimizer:
    """Reduces rollbacks in mixed conservative/optimistic mode.

    Parameters
    ----------
    num_nodes : int
        Number of nodes, each given a lookahead configuration.
    default_lookahead_ns : int
        The lookahead every node starts with.
    """

    def __init__(self, num_nodes: int, default_lookahead_ns: int = 1000):
        self.configs = {i: LookaheadConfig(i, default_lookahead_ns) for i in range(num_nodes)}

    def safe_advance_time(self, node_id: int) -> int:
        """Return the latest virtual time a node can reach without risking a rollback.

        Parameters
        ----------
        node_id : int
            The node that wants to advance.

        Returns
        -------
        int
            The smallest guarantee among the other nodes (last null message plus
            lookahead), or the node's own when it has no peers.

        Raises
        ------
        KeyError
            If the node has no configuration and no peers.
        """
        peers = [c for nid, c in self.configs.items() if nid != node_id]
        if not peers:
            return self.configs[node_id].last_null_message_ns + self.configs[node_id].lookahead_ns
        return min(c.last_null_message_ns + c.lookahead_ns for c in peers)

    def broadcast_null(self, node_id: int, current_ns: int) -> None:
        """Record a null message of one node at its current virtual time.

        Parameters
        ----------
        node_id : int
            The sending node.
        current_ns : int
            Its current virtual time.
        """
        self.configs[node_id].send_null_message(current_ns)


# ── Checkpoint Delta Encoding (Gap 2) ───────────────────────────────


@dataclass
class DeltaCheckpoint:
    """Stores only the diff from a base checkpoint.

    Attributes
    ----------
    base_checkpoint_id, checkpoint_id : int
        The checkpoint the diff starts from and the one it describes.
    virtual_time_ns : int
        Virtual time of the described state.
    node_id : int
        Node of the state.
    changed_indices : numpy.ndarray
        Indices along the first axis of the changed elements: flat indices for
        a one-dimensional state; for more dimensions a row index repeats once
        per changed element of that row.
    changed_values : numpy.ndarray
        The later state's entries at those indices (whole rows for a state of
        more than one dimension).
    lfsr_delta : int
        Change of the random-number-generator state; not set by
        :meth:`compute_delta`.
    size_bytes : int
        Bytes of the two arrays.
    """

    base_checkpoint_id: int
    checkpoint_id: int
    virtual_time_ns: int
    node_id: int
    changed_indices: np.ndarray[Any, Any]  # indices that changed
    changed_values: np.ndarray[Any, Any]  # new values at those indices
    lfsr_delta: int = 0
    size_bytes: int = 0

    @staticmethod
    def compute_delta(
        base_state: np.ndarray[Any, Any],
        new_state: np.ndarray[Any, Any],
        base_id: int,
        new_id: int,
        virtual_time_ns: int,
        node_id: int,
    ) -> DeltaCheckpoint:
        """Build the diff of two state arrays of the same shape.

        Parameters
        ----------
        base_state, new_state : numpy.ndarray
            The earlier and the later state; compared element by element.
        base_id, new_id : int
            Checkpoint ids of the two states.
        virtual_time_ns : int
            Virtual time of the later state.
        node_id : int
            Node of the states.

        Returns
        -------
        DeltaCheckpoint
            The indices and new values of the changed elements, along the first
            axis of the comparison.
        """
        diff_mask = base_state != new_state
        indices = np.where(diff_mask)[0]
        values = new_state[indices]
        return DeltaCheckpoint(
            base_checkpoint_id=base_id,
            checkpoint_id=new_id,
            virtual_time_ns=virtual_time_ns,
            node_id=node_id,
            changed_indices=indices,
            changed_values=values,
            size_bytes=indices.nbytes + values.nbytes,
        )

    @property
    def compression_ratio(self) -> float:
        """Placeholder ratio: 1.0 whenever the diff holds bytes, else 0.0.

        A real ratio needs the full state's size, which the diff does not keep.
        """
        if self.size_bytes <= 0:
            return 0.0
        return 1.0  # actual ratio requires full state size context

    @property
    def num_changes(self) -> int:
        """Number of changed elements in the diff."""
        return len(self.changed_indices)


# ── Replay Determinism Verifier (Gap 3) ─────────────────────────────


class ReplayVerifier:
    """Verifies bitstream-exact replay across runs.

    Compares checkpoint hashes from two runs to prove determinism.
    """

    def __init__(self) -> None:
        self.run_a_hashes: List[str] = []
        self.run_b_hashes: List[str] = []

    def record_run_a(self, checkpoint: Checkpoint) -> None:
        """Append a checkpoint's stored checksum to the first run.

        Parameters
        ----------
        checkpoint : Checkpoint
            A checkpoint whose checksum has been computed.
        """
        self.run_a_hashes.append(checkpoint.checksum)

    def record_run_b(self, checkpoint: Checkpoint) -> None:
        """Append a checkpoint's stored checksum to the second run.

        Parameters
        ----------
        checkpoint : Checkpoint
            A checkpoint whose checksum has been computed.
        """
        self.run_b_hashes.append(checkpoint.checksum)

    @property
    def is_deterministic(self) -> bool:
        """Whether both runs recorded checksums and agree over their common length.

        A shorter run that matches the longer one's beginning counts as
        deterministic.
        """
        if not self.run_a_hashes or not self.run_b_hashes:
            return False
        min_len = min(len(self.run_a_hashes), len(self.run_b_hashes))
        return self.run_a_hashes[:min_len] == self.run_b_hashes[:min_len]

    @property
    def first_divergence_index(self) -> Optional[int]:
        """Index of the first differing checksum over the common length, or None."""
        min_len = min(len(self.run_a_hashes), len(self.run_b_hashes))
        for i in range(min_len):
            if self.run_a_hashes[i] != self.run_b_hashes[i]:
                return i
        return None

    @property
    def compared_count(self) -> int:
        """Number of checksum pairs compared: the shorter run's length."""
        return min(len(self.run_a_hashes), len(self.run_b_hashes))


# ── Twin Drift Auto-Correction (Gap 4) ──────────────────────────────


@dataclass
class DriftCorrection:
    """One drift correction action.

    Attributes
    ----------
    correction_ns : int
        The time shift proposed for the digital twin.
    applied_at_ns : int
        The digital time at which the drift was seen.
    node_id : int
        The node concerned.
    reason : str
        The measured drift, for the record.
    """

    correction_ns: int
    applied_at_ns: int
    node_id: int
    reason: str


class DriftAutoCorrector:
    """Closed-loop drift correction between physical and digital twin.

    Parameters
    ----------
    max_drift_ns : int
        Drift tolerated without a correction.
    correction_gain : float
        Fraction of the drift the correction proposes.
    """

    def __init__(self, max_drift_ns: int = 5000, correction_gain: float = 0.5):
        self.max_drift_ns = max_drift_ns
        self.correction_gain = correction_gain
        self.corrections: List[DriftCorrection] = []

    def check_and_correct(
        self,
        physical_time_ns: int,
        digital_time_ns: int,
        node_id: int = 0,
    ) -> Optional[DriftCorrection]:
        """Propose a correction when the twin's time drifts past the tolerance.

        Nothing is applied; the correction is recorded and returned.

        Parameters
        ----------
        physical_time_ns, digital_time_ns : int
            The subject's and the twin's current times.
        node_id : int
            The node concerned.

        Returns
        -------
        DriftCorrection or None
            The gain times the drift, truncated to an integer, or None when the
            drift is within ``max_drift_ns``.
        """
        drift = physical_time_ns - digital_time_ns
        if abs(drift) <= self.max_drift_ns:
            return None
        correction = int(drift * self.correction_gain)
        dc = DriftCorrection(correction, digital_time_ns, node_id, f"drift={drift}ns")
        self.corrections.append(dc)
        return dc

    @property
    def total_corrections(self) -> int:
        """Number of corrections proposed so far."""
        return len(self.corrections)


# ── MPI Rank Topology Mapping (Gap 5) ───────────────────────────────


@dataclass
class MPIRankMapping:
    """Maps MPI ranks to physical node topology.

    Attributes
    ----------
    rank : int
        The MPI rank.
    hostname : str
        Host the rank runs on.
    gpu_id : int
        GPU of the rank, -1 for none.
    numa_node : int
        NUMA node of the rank.
    neuron_range : tuple of int
        Half-open range ``[first, end)`` of neuron ids the rank holds.
    """

    rank: int
    hostname: str = ""
    gpu_id: int = -1
    numa_node: int = 0
    neuron_range: Tuple[int, int] = (0, 0)

    @property
    def neuron_count(self) -> int:
        """Number of neurons in the rank's half-open range."""
        return self.neuron_range[1] - self.neuron_range[0]


class MPITopology:
    """Physical→logical node layout for distributed twin."""

    def __init__(self) -> None:
        self.ranks: Dict[int, MPIRankMapping] = {}

    def add_rank(self, mapping: MPIRankMapping) -> None:
        """Add or replace the mapping of one rank.

        Parameters
        ----------
        mapping : MPIRankMapping
            The rank's placement and neuron range.
        """
        self.ranks[mapping.rank] = mapping

    @property
    def total_neurons(self) -> int:
        """Sum of the neuron counts of all ranks."""
        return sum(r.neuron_count for r in self.ranks.values())

    @property
    def num_ranks(self) -> int:
        """Number of mapped ranks."""
        return len(self.ranks)

    def rank_for_neuron(self, neuron_id: int) -> Optional[int]:
        """Return the first mapped rank whose range holds a neuron.

        Parameters
        ----------
        neuron_id : int
            The neuron to place.

        Returns
        -------
        int or None
            The rank, in insertion order, or None when no range holds it.
        """
        for rank, m in self.ranks.items():
            if m.neuron_range[0] <= neuron_id < m.neuron_range[1]:
                return rank
        return None

    def co_located_ranks(self, rank: int) -> List[int]:
        """Return the other ranks on the same host as a rank, where communication is cheap.

        Parameters
        ----------
        rank : int
            The rank to look up.

        Returns
        -------
        list of int
            Other ranks with the same host name; empty for an unknown rank.
        """
        target = self.ranks.get(rank)
        if target is None:
            return []
        return [r for r, m in self.ranks.items() if m.hostname == target.hostname and r != rank]


# ── Event Rate Throttling / Backpressure (Gap 6) ────────────────────


class BackpressureController:
    """Prevents event overload by throttling injection rate.

    Parameters
    ----------
    max_queue_depth : int
        Queue depth at which offered events are refused.
    cooldown_ns : int
        Stored for the caller; the controller does not use it.
    """

    def __init__(self, max_queue_depth: int = 10000, cooldown_ns: int = 100):
        self.max_queue_depth = max_queue_depth
        self.cooldown_ns = cooldown_ns
        self.rejected_count: int = 0
        self.total_offered: int = 0

    def should_accept(self, current_queue_depth: int) -> bool:
        """Count an offered event and tell whether the queue has room for it.

        Parameters
        ----------
        current_queue_depth : int
            Events queued now.

        Returns
        -------
        bool
            False, and counted as a refusal, when the depth has reached
            ``max_queue_depth``.
        """
        self.total_offered += 1
        if current_queue_depth >= self.max_queue_depth:
            self.rejected_count += 1
            return False
        return True

    @property
    def rejection_rate(self) -> float:
        """Share of offered events refused; 0.0 before any offer."""
        if self.total_offered <= 0:
            return 0.0
        return self.rejected_count / self.total_offered

    @property
    def is_backpressured(self) -> bool:
        """Whether more than a tenth of the offered events were refused."""
        return self.rejection_rate > 0.1


# ── Checkpoint Integrity Audit Chain (Gap 7) ────────────────────────


class CheckpointAuditChain:
    """Tamper-evident chain of checkpoint hashes."""

    def __init__(self) -> None:
        self.chain: List[Tuple[int, str, str]] = []  # (cp_id, cp_hash, chain_hash)

    def append(self, checkpoint: Checkpoint) -> str:
        """Chain a checkpoint's stored checksum onto the previous link.

        Parameters
        ----------
        checkpoint : Checkpoint
            A checkpoint whose checksum has been computed.

        Returns
        -------
        str
            The new link: the SHA-256 of one text, the previous link (sixteen
            zeros before the first link) with the checkpoint's checksum appended,
            cut to its first 16 hexadecimal characters.
        """
        prev_hash = self.chain[-1][2] if self.chain else "0" * 16
        h = hashlib.sha256()
        h.update(prev_hash.encode())
        h.update(checkpoint.checksum.encode())
        chain_hash = h.hexdigest()[:16]
        self.chain.append((checkpoint.checkpoint_id, checkpoint.checksum, chain_hash))
        return chain_hash

    def verify(self) -> bool:
        """Recompute every link from the start and compare it with the stored one.

        Returns
        -------
        bool
            True when every stored link matches; an empty chain verifies.
        """
        prev = "0" * 16
        for cp_id, cp_hash, stored_chain_hash in self.chain:
            h = hashlib.sha256()
            h.update(prev.encode())
            h.update(cp_hash.encode())
            expected = h.hexdigest()[:16]
            if expected != stored_chain_hash:
                return False
            prev = stored_chain_hash
        return True

    @property
    def length(self) -> int:
        """Number of links in the chain."""
        return len(self.chain)


# ── Session Persistence (Gap 8) ─────────────────────────────────────


@dataclass
class SessionSnapshot:
    """Serialisable session state for persistence.

    Attributes
    ----------
    session_time_ns : int
        The session's time.
    num_nodes : int
        Nodes of the engine.
    mode : str
        The session's sync mode value.
    physical_events_in, digital_events_out : int
        The session's event counters.
    gvt_ns : int
        The engine's last computed global virtual time.
    total_rollbacks : int
        The engine's rollbacks.
    node_lvts : dict of int to int
        Local virtual time per node.
    checkpoint_count : int
        Checkpoints the engine holds.
    """

    session_time_ns: int
    num_nodes: int
    mode: str
    physical_events_in: int
    digital_events_out: int
    gvt_ns: int
    total_rollbacks: int
    node_lvts: Dict[int, int]
    checkpoint_count: int

    @staticmethod
    def from_session(session: TwinSession) -> SessionSnapshot:
        """Capture a session's counters and its engine's times.

        Parameters
        ----------
        session : TwinSession
            The session to capture.

        Returns
        -------
        SessionSnapshot
            The snapshot; the global virtual time is the engine's last computed one.
        """
        eng = session.engine
        return SessionSnapshot(
            session_time_ns=session.session_time_ns,
            num_nodes=session.num_nodes,
            mode=session.mode.value,
            physical_events_in=session.physical_events_in,
            digital_events_out=session.digital_events_out,
            gvt_ns=eng.gvt_ns,
            total_rollbacks=eng.total_rollbacks,
            node_lvts={nid: n.local_virtual_time_ns for nid, n in eng.nodes.items()},
            checkpoint_count=eng.checkpoint_mgr.total_checkpoints,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Return the snapshot as a dictionary of its fields.

        Returns
        -------
        dict
            One key per field, with the same values.
        """
        return {
            "session_time_ns": self.session_time_ns,
            "num_nodes": self.num_nodes,
            "mode": self.mode,
            "physical_events_in": self.physical_events_in,
            "digital_events_out": self.digital_events_out,
            "gvt_ns": self.gvt_ns,
            "total_rollbacks": self.total_rollbacks,
            "node_lvts": self.node_lvts,
            "checkpoint_count": self.checkpoint_count,
        }


# ── Multi-Twin Federation (Gap 9) ───────────────────────────────────


@dataclass
class TwinEndpoint:
    """One twin in a federation.

    Attributes
    ----------
    twin_id : str
        The twin's name in the federation.
    session : TwinSession
        The twin's session.
    priority : int
        Stored for the caller; the federation does not order by it.
    """

    twin_id: str
    session: TwinSession
    priority: int = 0


class TwinFederation:
    """Federates multiple digital twins for multi-subject studies."""

    def __init__(self) -> None:
        self.twins: Dict[str, TwinEndpoint] = {}

    def register(self, twin_id: str, session: TwinSession, priority: int = 0) -> None:
        """Add or replace a twin under a name.

        Parameters
        ----------
        twin_id : str
            The twin's name.
        session : TwinSession
            The twin's session.
        priority : int
            Stored with the twin.
        """
        self.twins[twin_id] = TwinEndpoint(twin_id, session, priority)

    @property
    def twin_count(self) -> int:
        """Number of registered twins."""
        return len(self.twins)

    def global_gvt(self) -> int:
        """Return the smallest last computed global virtual time of all twins.

        Returns
        -------
        int
            The minimum, or 0 for an empty federation.
        """
        if not self.twins:
            return 0
        return min(t.session.engine.gvt_ns for t in self.twins.values())

    def advance_all(self, steps: int = 1) -> Dict[str, int]:
        """Advance every twin's session by up to a number of events.

        Parameters
        ----------
        steps : int
            Most events each twin processes.

        Returns
        -------
        dict of str to int
            Events processed per twin.
        """
        return {tid: t.session.advance(steps) for tid, t in self.twins.items()}

    def total_divergence(self) -> float:
        """Return the sum of the twins' total divergences.

        Returns
        -------
        float
            The sum, or 0.0 for an empty federation.
        """
        if not self.twins:
            return 0.0
        return sum(t.session.divergence.total_divergence for t in self.twins.values())


# ── Adaptive Checkpoint Interval (Gap 10) ───────────────────────────


class AdaptiveCheckpointInterval:
    """Dynamically adjusts checkpoint frequency based on rollback rate.

    Parameters
    ----------
    base_interval : int
        The starting interval.
    min_interval, max_interval : int
        Bounds the interval is kept within when it changes.
    """

    def __init__(
        self, base_interval: int = 1000, min_interval: int = 100, max_interval: int = 10000
    ):
        self.base_interval = base_interval
        self.min_interval = min_interval
        self.max_interval = max_interval
        self.current_interval = base_interval
        self._last_rollbacks: int = 0

    def update(self, total_rollbacks: int, total_events: int) -> int:
        """Halve or double the interval from the rollbacks since the last update.

        Parameters
        ----------
        total_rollbacks : int
            Cumulative rollbacks so far.
        total_events : int
            Events to relate the new rollbacks to.

        Returns
        -------
        int
            The interval: halved, not below ``min_interval``, when the new
            rollbacks exceed 5 % of the events; doubled, not above
            ``max_interval``, below 1 %; unchanged otherwise or without events.
        """
        new_rollbacks = total_rollbacks - self._last_rollbacks
        self._last_rollbacks = total_rollbacks

        if total_events <= 0:
            return self.current_interval

        rollback_rate = new_rollbacks / max(1, total_events)
        if rollback_rate > 0.05:
            self.current_interval = max(self.min_interval, self.current_interval // 2)
        elif rollback_rate < 0.01:
            self.current_interval = min(self.max_interval, self.current_interval * 2)

        return self.current_interval

    @property
    def is_aggressive(self) -> bool:
        """Whether the interval is at most twice the minimum."""
        return self.current_interval <= self.min_interval * 2

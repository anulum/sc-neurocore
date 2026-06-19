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
    """Lamport logical clock for causal ordering."""

    def __init__(self) -> None:
        self.time: int = 0

    def tick(self) -> int:
        """Local event: increment."""
        self.time += 1
        return self.time

    def send(self) -> int:
        """Prepare timestamp for sending."""
        self.time += 1
        return self.time

    def receive(self, remote_time: int) -> int:
        """Update on message receipt."""
        self.time = max(self.time, remote_time) + 1
        return self.time


class VectorClock:
    """Vector clock for full causal dependency tracking."""

    def __init__(self, node_id: int, num_nodes: int):
        self.node_id = node_id
        self.num_nodes = num_nodes
        self.clock = np.zeros(num_nodes, dtype=np.int64)

    def tick(self) -> np.ndarray[Any, Any]:
        self.clock[self.node_id] += 1
        return self.clock.copy()

    def send(self) -> np.ndarray[Any, Any]:
        self.clock[self.node_id] += 1
        return self.clock.copy()

    def receive(self, remote_clock: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        self.clock = np.maximum(self.clock, remote_clock)
        self.clock[self.node_id] += 1
        return self.clock.copy()

    def happened_before(self, other: np.ndarray[Any, Any]) -> bool:
        """Check if self happened-before other (self < other)."""
        return bool(np.all(self.clock <= other) and np.any(self.clock < other))

    def concurrent_with(self, other: np.ndarray[Any, Any]) -> bool:
        """Check if self is concurrent with other."""
        return not self.happened_before(other) and not bool(
            np.all(other <= self.clock) and np.any(other < self.clock)
        )


# ── Events ───────────────────────────────────────────────────────────


class EventType(Enum):
    SPIKE = "spike"
    SENSOR_INPUT = "sensor_input"
    CHECKPOINT = "checkpoint"
    ANTI_MESSAGE = "anti_message"
    ROLLBACK = "rollback"
    SYNC_BARRIER = "sync_barrier"


@dataclass(order=True)
class TwinEvent:
    """One event in the time-warp simulation."""

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
    """Deterministic state snapshot for rollback."""

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
        """Find the latest checkpoint at or before target_time."""
        cps = self.checkpoints.get(node_id, [])
        best = None
        for cp in cps:
            if cp.virtual_time_ns <= target_time_ns:
                best = cp
        return best

    def discard_after(self, node_id: int, time_ns: int) -> int:
        """Discard checkpoints after a given time (post-rollback cleanup)."""
        cps = self.checkpoints.get(node_id, [])
        before = len(cps)
        self.checkpoints[node_id] = [cp for cp in cps if cp.virtual_time_ns <= time_ns]
        return before - len(self.checkpoints.get(node_id, []))

    @property
    def total_checkpoints(self) -> int:
        return sum(len(v) for v in self.checkpoints.values())


# ── Time-Warp Engine ────────────────────────────────────────────────


@dataclass
class NodeState:
    """Per-node state in the time-warp simulation."""

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
        """Inject an event into the simulation."""
        heapq.heappush(self.event_queue, event)

    def process_next(self) -> Optional[TwinEvent]:
        """Process the next event from the queue."""
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
        """Compute Global Virtual Time (minimum of all LVTs + in-transit)."""
        lvts = [n.local_virtual_time_ns for n in self.nodes.values()]
        in_transit = [e.virtual_time_ns for e in self.event_queue if not e.cancelled]
        all_times = lvts + in_transit
        self.gvt_ns = min(all_times) if all_times else 0
        return self.gvt_ns

    def fossil_collect(self) -> int:
        """Remove checkpoints below GVT."""
        gvt = self.compute_gvt()
        removed = 0
        for nid in list(self.checkpoint_mgr.checkpoints.keys()):
            cps = self.checkpoint_mgr.checkpoints[nid]
            before = len(cps)
            self.checkpoint_mgr.checkpoints[nid] = [cp for cp in cps if cp.virtual_time_ns >= gvt]
            removed += before - len(self.checkpoint_mgr.checkpoints[nid])
        return removed

    def status(self) -> Dict[str, Any]:
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
        """Inject a sync barrier event to all nodes at given time."""
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
        """Verify causal ordering of processed events.

        Returns list of (index_a, index_b) pairs where order is violated.
        """
        violations = []
        for i in range(len(self.processed) - 1):
            a = self.processed[i]
            b = self.processed[i + 1]
            if a.target_node == b.target_node and a.virtual_time_ns > b.virtual_time_ns:
                violations.append((i, i + 1))
        return violations

    def detect_starvation(self, threshold_ns: int = 10000) -> List[int]:
        """Detect nodes lagging behind GVT by more than threshold."""
        gvt = self.compute_gvt()
        return [
            nid for nid, n in self.nodes.items() if gvt - n.local_virtual_time_ns > threshold_ns
        ]

    def node_throughput(self) -> Dict[int, int]:
        """Events processed per node."""
        return {nid: n.processed_events for nid, n in self.nodes.items()}


# ── Twin Session ────────────────────────────────────────────────────


class SyncMode(Enum):
    LOCKSTEP = "lockstep"
    OPTIMISTIC = "optimistic"
    CONSERVATIVE = "conservative"


@dataclass
class DivergenceMetric:
    """Measures divergence between physical and digital twin."""

    spike_rate_divergence: float = 0.0
    timing_offset_ns: int = 0
    identity_drift: float = 0.0
    causal_violations: int = 0

    @property
    def total_divergence(self) -> float:
        return (
            self.spike_rate_divergence
            + abs(self.timing_offset_ns) / 1e6
            + self.identity_drift
            + self.causal_violations * 0.1
        )

    @property
    def within_tolerance(self) -> bool:
        return self.total_divergence < 1.0


class TwinSession:
    """Orchestrates physical ↔ digital twin synchronization.

    Manages bidirectional data flow:
    - Physical → Digital: sensor events (MEA spikes, EEG)
    - Digital → Physical: stimulation commands (opto, TMS)
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
        self.running = True

    def stop(self) -> None:
        self.running = False

    def inject_physical_event(
        self, spike_time_ns: int, neuron_id: int, target_node: int = 0
    ) -> None:
        """Inject a physical sensor event into the digital twin."""
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
        """Advance the simulation by N steps."""
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
        """Update divergence metrics."""
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
    """

    node_id: int
    lookahead_ns: int = 1000
    last_null_message_ns: int = 0

    def can_advance_to(self, target_ns: int) -> bool:
        return target_ns <= self.last_null_message_ns + self.lookahead_ns

    def send_null_message(self, current_ns: int) -> int:
        self.last_null_message_ns = current_ns
        return current_ns + self.lookahead_ns


class NullMessageOptimizer:
    """Reduces rollbacks in mixed conservative/optimistic mode."""

    def __init__(self, num_nodes: int, default_lookahead_ns: int = 1000):
        self.configs = {i: LookaheadConfig(i, default_lookahead_ns) for i in range(num_nodes)}

    def safe_advance_time(self, node_id: int) -> int:
        """Maximum time this node can safely advance to."""
        peers = [c for nid, c in self.configs.items() if nid != node_id]
        if not peers:
            return self.configs[node_id].last_null_message_ns + self.configs[node_id].lookahead_ns
        return min(c.last_null_message_ns + c.lookahead_ns for c in peers)

    def broadcast_null(self, node_id: int, current_ns: int) -> None:
        self.configs[node_id].send_null_message(current_ns)


# ── Checkpoint Delta Encoding (Gap 2) ───────────────────────────────


@dataclass
class DeltaCheckpoint:
    """Stores only the diff from a base checkpoint."""

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
        if self.size_bytes <= 0:
            return 0.0
        return 1.0  # actual ratio requires full state size context

    @property
    def num_changes(self) -> int:
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
        self.run_a_hashes.append(checkpoint.checksum)

    def record_run_b(self, checkpoint: Checkpoint) -> None:
        self.run_b_hashes.append(checkpoint.checksum)

    @property
    def is_deterministic(self) -> bool:
        if not self.run_a_hashes or not self.run_b_hashes:
            return False
        min_len = min(len(self.run_a_hashes), len(self.run_b_hashes))
        return self.run_a_hashes[:min_len] == self.run_b_hashes[:min_len]

    @property
    def first_divergence_index(self) -> Optional[int]:
        min_len = min(len(self.run_a_hashes), len(self.run_b_hashes))
        for i in range(min_len):
            if self.run_a_hashes[i] != self.run_b_hashes[i]:
                return i
        return None

    @property
    def compared_count(self) -> int:
        return min(len(self.run_a_hashes), len(self.run_b_hashes))


# ── Twin Drift Auto-Correction (Gap 4) ──────────────────────────────


@dataclass
class DriftCorrection:
    """One drift correction action."""

    correction_ns: int
    applied_at_ns: int
    node_id: int
    reason: str


class DriftAutoCorrector:
    """Closed-loop drift correction between physical and digital twin."""

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
        drift = physical_time_ns - digital_time_ns
        if abs(drift) <= self.max_drift_ns:
            return None
        correction = int(drift * self.correction_gain)
        dc = DriftCorrection(correction, digital_time_ns, node_id, f"drift={drift}ns")
        self.corrections.append(dc)
        return dc

    @property
    def total_corrections(self) -> int:
        return len(self.corrections)


# ── MPI Rank Topology Mapping (Gap 5) ───────────────────────────────


@dataclass
class MPIRankMapping:
    """Maps MPI ranks to physical node topology."""

    rank: int
    hostname: str = ""
    gpu_id: int = -1
    numa_node: int = 0
    neuron_range: Tuple[int, int] = (0, 0)

    @property
    def neuron_count(self) -> int:
        return self.neuron_range[1] - self.neuron_range[0]


class MPITopology:
    """Physical→logical node layout for distributed twin."""

    def __init__(self) -> None:
        self.ranks: Dict[int, MPIRankMapping] = {}

    def add_rank(self, mapping: MPIRankMapping) -> None:
        self.ranks[mapping.rank] = mapping

    @property
    def total_neurons(self) -> int:
        return sum(r.neuron_count for r in self.ranks.values())

    @property
    def num_ranks(self) -> int:
        return len(self.ranks)

    def rank_for_neuron(self, neuron_id: int) -> Optional[int]:
        for rank, m in self.ranks.items():
            if m.neuron_range[0] <= neuron_id < m.neuron_range[1]:
                return rank
        return None

    def co_located_ranks(self, rank: int) -> List[int]:
        """Ranks on the same host (cheap communication)."""
        target = self.ranks.get(rank)
        if target is None:
            return []
        return [r for r, m in self.ranks.items() if m.hostname == target.hostname and r != rank]


# ── Event Rate Throttling / Backpressure (Gap 6) ────────────────────


class BackpressureController:
    """Prevents event overload by throttling injection rate."""

    def __init__(self, max_queue_depth: int = 10000, cooldown_ns: int = 100):
        self.max_queue_depth = max_queue_depth
        self.cooldown_ns = cooldown_ns
        self.rejected_count: int = 0
        self.total_offered: int = 0

    def should_accept(self, current_queue_depth: int) -> bool:
        self.total_offered += 1
        if current_queue_depth >= self.max_queue_depth:
            self.rejected_count += 1
            return False
        return True

    @property
    def rejection_rate(self) -> float:
        if self.total_offered <= 0:
            return 0.0
        return self.rejected_count / self.total_offered

    @property
    def is_backpressured(self) -> bool:
        return self.rejection_rate > 0.1


# ── Checkpoint Integrity Audit Chain (Gap 7) ────────────────────────


class CheckpointAuditChain:
    """Tamper-evident chain of checkpoint hashes."""

    def __init__(self) -> None:
        self.chain: List[Tuple[int, str, str]] = []  # (cp_id, cp_hash, chain_hash)

    def append(self, checkpoint: Checkpoint) -> str:
        prev_hash = self.chain[-1][2] if self.chain else "0" * 16
        h = hashlib.sha256()
        h.update(prev_hash.encode())
        h.update(checkpoint.checksum.encode())
        chain_hash = h.hexdigest()[:16]
        self.chain.append((checkpoint.checkpoint_id, checkpoint.checksum, chain_hash))
        return chain_hash

    def verify(self) -> bool:
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
        return len(self.chain)


# ── Session Persistence (Gap 8) ─────────────────────────────────────


@dataclass
class SessionSnapshot:
    """Serializable session state for persistence."""

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
    """One twin in a federation."""

    twin_id: str
    session: TwinSession
    priority: int = 0


class TwinFederation:
    """Federates multiple digital twins for multi-subject studies."""

    def __init__(self) -> None:
        self.twins: Dict[str, TwinEndpoint] = {}

    def register(self, twin_id: str, session: TwinSession, priority: int = 0) -> None:
        self.twins[twin_id] = TwinEndpoint(twin_id, session, priority)

    @property
    def twin_count(self) -> int:
        return len(self.twins)

    def global_gvt(self) -> int:
        if not self.twins:
            return 0
        return min(t.session.engine.gvt_ns for t in self.twins.values())

    def advance_all(self, steps: int = 1) -> Dict[str, int]:
        return {tid: t.session.advance(steps) for tid, t in self.twins.items()}

    def total_divergence(self) -> float:
        if not self.twins:
            return 0.0
        return sum(t.session.divergence.total_divergence for t in self.twins.values())


# ── Adaptive Checkpoint Interval (Gap 10) ───────────────────────────


class AdaptiveCheckpointInterval:
    """Dynamically adjusts checkpoint frequency based on rollback rate."""

    def __init__(
        self, base_interval: int = 1000, min_interval: int = 100, max_interval: int = 10000
    ):
        self.base_interval = base_interval
        self.min_interval = min_interval
        self.max_interval = max_interval
        self.current_interval = base_interval
        self._last_rollbacks: int = 0

    def update(self, total_rollbacks: int, total_events: int) -> int:
        """Adjust interval: more rollbacks → more frequent checkpoints."""
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
        return self.current_interval <= self.min_interval * 2

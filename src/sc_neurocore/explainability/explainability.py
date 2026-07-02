# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Traceable Stochastic Explainability Layer

"""Bitstream-level explainability with deterministic replay and provenance.

Leverages bit-true deterministic SC streams and LFSR replay to generate
full provenance traces: "why this spike fired" as a verifiable bitstream
decision tree.  Integrates with the ``predictive_coding.ReasoningTrace``
for neuro-symbolic explanations and produces hash-verified audit trails.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


_Q16_MIN = 0
_Q16_MAX = 0xFFFF


def _validate_replay_parameters(
    *,
    threshold_q16: int,
    bitstream_length: int,
    spike_threshold_count: Optional[int] = None,
    length_name: str = "bitstream_length",
) -> None:
    """Validate deterministic replay parameters before replay side effects."""
    if isinstance(threshold_q16, bool) or not _Q16_MIN <= threshold_q16 <= _Q16_MAX:
        raise ValueError("threshold_q16 must be an integer in [0, 65535]")
    if isinstance(bitstream_length, bool) or bitstream_length <= 0:
        raise ValueError(f"{length_name} must be positive")
    if spike_threshold_count is None:
        return
    if (
        isinstance(spike_threshold_count, bool)
        or spike_threshold_count < 0
        or spike_threshold_count > bitstream_length
    ):
        raise ValueError("spike_threshold_count must be between 0 and bitstream_length")


# ── LFSR Replay Engine ───────────────────────────────────────────────


class LFSRReplay:
    """Deterministic LFSR-16 replay engine.

    Mirrors ``core_engine::Lfsr16`` polynomial: x^16 + x^14 + x^13 + x^11 + 1.
    Given the same seed, reproduces the exact same bitstream for formal
    verification and replay-based auditing.
    """

    def __init__(self, seed: int) -> None:
        if seed == 0:
            raise ValueError("LFSR seed must be non-zero")
        self.initial_seed = seed
        self.reg = seed & 0xFFFF

    def step(self) -> int:
        """Advance one step, return new register value."""
        feedback = ((self.reg >> 15) ^ (self.reg >> 13) ^ (self.reg >> 12) ^ (self.reg >> 10)) & 1
        self.reg = ((self.reg << 1) | feedback) & 0xFFFF
        return self.reg

    def encode(self, threshold: int, length: int) -> np.ndarray[Any, Any]:
        """Generate a bitstream by comparing LFSR output against threshold."""
        bits = np.zeros(length, dtype=np.uint8)
        for i in range(length):
            bits[i] = 1 if self.reg < threshold else 0
            self.step()
        return bits

    def reset(self) -> None:
        """Reset to initial seed for replay."""
        self.reg = self.initial_seed


# ── Decision Tree ────────────────────────────────────────────────────


class SpikeDecision(Enum):
    SPIKE = "spike"
    NO_SPIKE = "no_spike"
    UNDETERMINED = "undetermined"


@dataclass
class DecisionMargin:
    """How close a decision was to flipping."""

    popcount: int
    threshold: int
    margin: int  # popcount - threshold (negative = no spike)
    confidence: float  # |margin| / bitstream_length


@dataclass
class DecisionNode:
    """One node in a spike decision tree."""

    neuron_id: str
    popcount: int
    threshold: int
    bitstream_length: int
    probability: float
    scc_context: float = 0.0
    scc_influence: float = 0.0
    decision: SpikeDecision = SpikeDecision.UNDETERMINED
    children: List[DecisionNode] = field(default_factory=list)
    bitstream_hash: str = ""
    timestep: int = 0
    threshold_q16: int = 0
    layer_id: str = ""
    contributing_neurons: List[str] = field(default_factory=list)

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    @property
    def margin(self) -> DecisionMargin:
        m = self.popcount - self.threshold
        conf = abs(m) / self.bitstream_length if self.bitstream_length > 0 else 0.0
        return DecisionMargin(self.popcount, self.threshold, m, conf)


class SpikeDecisionTree:
    """Captures "why this spike fired" as a verifiable decision tree."""

    def __init__(self) -> None:
        self.root: Optional[DecisionNode] = None
        self._nodes: List[DecisionNode] = []

    def add_decision(
        self,
        neuron_id: str,
        bitstream: np.ndarray[Any, Any],
        threshold: int,
        scc: float = 0.0,
        parent: Optional[DecisionNode] = None,
        timestep: int = 0,
        layer_id: str = "",
        contributing_neurons: Optional[List[str]] = None,
        threshold_q16: int = 0,
    ) -> DecisionNode:
        """Record a spike decision from bitstream observation."""
        popcount = int(np.sum(bitstream))
        length = len(bitstream)
        prob = popcount / length if length > 0 else 0.0
        decision = SpikeDecision.SPIKE if popcount >= threshold else SpikeDecision.NO_SPIKE
        bs_hash = hashlib.sha256(bitstream.tobytes()).hexdigest()[:16]

        scc_influence = abs(scc) * (popcount / max(length, 1))

        node = DecisionNode(
            neuron_id=neuron_id,
            popcount=popcount,
            threshold=threshold,
            bitstream_length=length,
            probability=prob,
            scc_context=scc,
            scc_influence=scc_influence,
            decision=decision,
            bitstream_hash=bs_hash,
            timestep=timestep,
            layer_id=layer_id,
            contributing_neurons=contributing_neurons or [],
            threshold_q16=threshold_q16,
        )

        if parent is not None:
            parent.children.append(node)
        elif self.root is None:
            self.root = node

        self._nodes.append(node)
        return node

    @property
    def depth(self) -> int:
        if self.root is None:
            return 0
        return self._compute_depth(self.root)

    def _compute_depth(self, node: DecisionNode) -> int:
        if not node.children:
            return 1
        return 1 + max(self._compute_depth(c) for c in node.children)

    @property
    def num_spikes(self) -> int:
        return sum(1 for n in self._nodes if n.decision == SpikeDecision.SPIKE)

    @property
    def num_nodes(self) -> int:
        return len(self._nodes)

    def nodes_at_layer(self, layer_id: str) -> List[DecisionNode]:
        """Return all nodes at a given layer."""
        return [n for n in self._nodes if n.layer_id == layer_id]

    def nodes_at_timestep(self, timestep: int) -> List[DecisionNode]:
        """Return all nodes at a given timestep."""
        return [n for n in self._nodes if n.timestep == timestep]

    def get_node(self, neuron_id: str) -> Optional[DecisionNode]:
        """Look up a node by neuron ID."""
        for n in self._nodes:
            if n.neuron_id == neuron_id:
                return n
        return None

    def spike_path(self) -> List[DecisionNode]:
        """Return the chain of spiking nodes from root down."""
        if self.root is None:
            return []
        path: List[DecisionNode] = []
        self._collect_spike_path(self.root, path)
        return path

    def _collect_spike_path(self, node: DecisionNode, path: List[DecisionNode]) -> None:
        if node.decision == SpikeDecision.SPIKE:
            path.append(node)
        for c in node.children:
            self._collect_spike_path(c, path)

    def to_dict(self) -> Dict[str, Any]:
        if self.root is None:
            return {}
        return self._node_to_dict(self.root)

    def _node_to_dict(self, node: DecisionNode) -> Dict[str, Any]:
        return {
            "neuron_id": node.neuron_id,
            "popcount": node.popcount,
            "threshold": node.threshold,
            "probability": node.probability,
            "decision": node.decision.value,
            "bitstream_hash": node.bitstream_hash,
            "scc_influence": node.scc_influence,
            "margin": node.margin.margin,
            "confidence": node.margin.confidence,
            "timestep": node.timestep,
            "layer_id": node.layer_id,
            "contributing_neurons": node.contributing_neurons,
            "children": [self._node_to_dict(c) for c in node.children],
        }


# ── Provenance Trace ─────────────────────────────────────────────────


@dataclass
class ProvenanceStep:
    """One step in a full provenance chain."""

    stage: str
    description: str
    data_hash: str
    timestamp_ns: int
    metadata: Dict[str, Any] = field(default_factory=dict)


class ProvenanceTrace:
    """Full chain from input → encoding → computation → spike decision."""

    def __init__(self) -> None:
        self._steps: List[ProvenanceStep] = []
        self._complete = False

    def add_step(
        self,
        stage: str,
        description: str,
        data: Optional[np.ndarray[Any, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ProvenanceStep:
        """Record one provenance step."""
        if data is not None:
            data_hash = hashlib.sha256(data.tobytes()).hexdigest()[:16]
        else:
            data_hash = hashlib.sha256(description.encode()).hexdigest()[:16]

        step = ProvenanceStep(
            stage=stage,
            description=description,
            data_hash=data_hash,
            timestamp_ns=time.perf_counter_ns(),
            metadata=metadata or {},
        )
        self._steps.append(step)
        return step

    def finalize(self) -> None:
        self._complete = True

    @property
    def is_complete(self) -> bool:
        return self._complete

    @property
    def num_steps(self) -> int:
        return len(self._steps)

    @property
    def chain_hash(self) -> str:
        """Hash of the entire provenance chain for tamper detection."""
        h = hashlib.sha256()
        for step in self._steps:
            h.update(step.data_hash.encode())
            h.update(step.stage.encode())
        return h.hexdigest()

    def to_list(self) -> List[Dict[str, Any]]:
        return [
            {
                "stage": s.stage,
                "description": s.description,
                "data_hash": s.data_hash,
                "timestamp_ns": s.timestamp_ns,
                "metadata": s.metadata,
            }
            for s in self._steps
        ]


# ── Verifiability ────────────────────────────────────────────────────


@dataclass
class RegulatoryMetadata:
    """IEC 62304 / FDA SaMD traceability fields."""

    device_class: str = "Class II"
    risk_level: str = "moderate"
    intended_use: str = ""
    software_version: str = ""
    udi: str = ""  # Unique Device Identifier
    sudi_hash: str = ""  # Hash of the software + config
    operator_id: str = ""
    review_status: str = "pending"


@dataclass
class FormalPropertyLink:
    """Cross-reference to SymbiYosys formal verification."""

    property_name: str
    property_file: str = ""
    status: str = "unverified"  # proven / cex / unverified
    bounded_depth: int = 0
    engine: str = "sby"


@dataclass
class VerifiabilityReport:
    """Formal audit report with hash verification."""

    chain_hash: str
    num_steps: int
    all_hashes_valid: bool
    decision_tree_depth: int
    total_spikes: int
    replay_seed: int
    replay_matches: bool
    regulatory: Optional[RegulatoryMetadata] = None
    formal_properties: List[FormalPropertyLink] = field(default_factory=list)


# ── Sensitivity Analysis ─────────────────────────────────────────────


@dataclass
class SensitivityResult:
    """Result of a 'what-if' threshold perturbation."""

    neuron_id: str
    original_threshold: int
    perturbed_threshold: int
    original_decision: SpikeDecision
    perturbed_decision: SpikeDecision
    flipped: bool


class SensitivityAnalyzer:
    """Counterfactual analysis: 'would the decision flip if threshold ±N?'"""

    @staticmethod
    def analyze(
        node: DecisionNode,
        perturbations: Optional[List[int]] = None,
    ) -> List[SensitivityResult]:
        if perturbations is None:
            perturbations = [-10, -5, -1, 1, 5, 10]
        results = []
        for delta in perturbations:
            new_t = max(0, node.threshold + delta)
            new_dec = SpikeDecision.SPIKE if node.popcount >= new_t else SpikeDecision.NO_SPIKE
            results.append(
                SensitivityResult(
                    neuron_id=node.neuron_id,
                    original_threshold=node.threshold,
                    perturbed_threshold=new_t,
                    original_decision=node.decision,
                    perturbed_decision=new_dec,
                    flipped=(new_dec != node.decision),
                )
            )
        return results

    @staticmethod
    def critical_delta(node: DecisionNode) -> int:
        """Smallest threshold change that flips the decision."""
        m = node.margin
        if m.margin >= 0:
            return m.margin + 1
        return m.margin


# ── Causal Attribution ───────────────────────────────────────────────


@dataclass
class CausalAttribution:
    """Attribution of a spike to upstream neurons."""

    target_neuron: str
    attributions: Dict[str, float]  # source_neuron -> contribution weight
    total_contribution: float

    @property
    def top_contributors(self) -> List[Tuple[str, float]]:
        """Sorted (descending) list of contributing neurons."""
        return sorted(self.attributions.items(), key=lambda x: x[1], reverse=True)


class CausalAttributor:
    """Computes causal attribution from input neurons to output spike."""

    @staticmethod
    def attribute(
        target: DecisionNode,
        input_bitstreams: Dict[str, np.ndarray[Any, Any]],
        weights: Optional[Dict[str, float]] = None,
    ) -> CausalAttribution:
        """Compute per-input-neuron contribution to the target popcount."""
        attribs: Dict[str, float] = {}
        for nid, bs in input_bitstreams.items():
            w = weights.get(nid, 1.0) if weights else 1.0
            contribution = float(np.sum(bs)) * w
            attribs[nid] = contribution
        total = sum(attribs.values())
        return CausalAttribution(
            target_neuron=target.neuron_id,
            attributions=attribs,
            total_contribution=total,
        )


# ── Explanation Diff ─────────────────────────────────────────────────


@dataclass
class DiffEntry:
    """One field that differs between two explanations."""

    field: str
    value_a: Any
    value_b: Any


class ExplanationDiff:
    """Compares two decision nodes to find divergence points."""

    @staticmethod
    def diff(a: DecisionNode, b: DecisionNode) -> List[DiffEntry]:
        diffs = []
        for attr in [
            "neuron_id",
            "popcount",
            "threshold",
            "bitstream_length",
            "probability",
            "scc_context",
            "decision",
            "bitstream_hash",
        ]:
            va = getattr(a, attr)
            vb = getattr(b, attr)
            if va != vb:
                diffs.append(DiffEntry(attr, va, vb))
        return diffs


# ── Temporal Windowing ───────────────────────────────────────────────


class TemporalWindow:
    """Records decisions across timesteps for temporal attribution."""

    def __init__(self) -> None:
        self._windows: Dict[int, List[DecisionNode]] = {}

    def add(self, node: DecisionNode) -> None:
        self._windows.setdefault(node.timestep, []).append(node)

    def spike_rate_at(self, timestep: int) -> float:
        nodes = self._windows.get(timestep, [])
        if not nodes:
            return 0.0
        return sum(1 for n in nodes if n.decision == SpikeDecision.SPIKE) / len(nodes)

    def active_timesteps(self) -> List[int]:
        return sorted(self._windows.keys())

    def peak_timestep(self) -> int:
        """Timestep with the highest spike rate."""
        best_t = 0
        best_rate = -1.0
        for t in self._windows:
            rate = self.spike_rate_at(t)
            if rate > best_rate:
                best_rate = rate
                best_t = t
        return best_t

    @property
    def num_timesteps(self) -> int:
        return len(self._windows)


# ── Natural Language Explainer ───────────────────────────────────────


class NaturalLanguageExplainer:
    """Generates human-readable explanation strings."""

    @staticmethod
    def explain_node(node: DecisionNode) -> str:
        m = node.margin
        if node.decision == SpikeDecision.SPIKE:
            desc = (
                f"Neuron {node.neuron_id} fired at timestep {node.timestep}. "
                f"Popcount {node.popcount} exceeded threshold {node.threshold} "
                f"by {m.margin} bits (confidence {m.confidence:.1%}). "
                f"Encoded probability was {node.probability:.3f}."
            )
        else:
            desc = (
                f"Neuron {node.neuron_id} did NOT fire at timestep {node.timestep}. "
                f"Popcount {node.popcount} fell short of threshold {node.threshold} "
                f"by {abs(m.margin)} bits. "
                f"Encoded probability was {node.probability:.3f}."
            )
        if node.scc_context > 0:
            desc += f" Correlation context SCC={node.scc_context:.3f} may have biased encoding."
        if node.contributing_neurons:
            desc += f" Driven by inputs: {', '.join(node.contributing_neurons[:5])}."
        return desc

    @staticmethod
    def explain_attribution(attr: CausalAttribution) -> str:
        top = attr.top_contributors[:3]
        parts = [f"{nid} ({w:.1f})" for nid, w in top]
        return (
            f"Spike at {attr.target_neuron} was primarily caused by: "
            f"{', '.join(parts)}. Total input contribution: {attr.total_contribution:.1f}."
        )

    @staticmethod
    def explain_sensitivity(results: List[SensitivityResult]) -> str:
        flips = [r for r in results if r.flipped]
        if not flips:
            return "Decision is robust to all tested perturbations."
        smallest = min(flips, key=lambda r: abs(r.perturbed_threshold - r.original_threshold))
        return (
            f"Decision would flip if threshold changed by "
            f"{smallest.perturbed_threshold - smallest.original_threshold:+d} "
            f"(from {smallest.original_threshold} to {smallest.perturbed_threshold})."
        )

    @staticmethod
    def enhance_with_local_llm(
        text: str,
        *,
        bridge: Any,
        question: str = (
            "Rewrite this deterministic spike explanation for a human operator. "
            "Preserve all numeric facts and keep the wording concise."
        ),
    ) -> str:
        """Enhance a deterministic explanation through the local LLM bridge.

        The caller must supply a configured ``LocalLLMBridge`` instance so this
        module keeps no hard runtime dependency on any local model server.
        """
        response = bridge.chat(f"{question}\n\nBase explanation:\n{text}")
        enhanced: str = response.text
        return enhanced

    @staticmethod
    def explain_node_with_local_llm(
        node: DecisionNode,
        *,
        bridge: Any,
        question: str = (
            "Explain this spike decision for a human operator in two short paragraphs. "
            "Do not change or invent numeric values."
        ),
    ) -> str:
        """Generate a local-LLM-enhanced explanation for one decision node."""
        base = NaturalLanguageExplainer.explain_node(node)
        return NaturalLanguageExplainer.enhance_with_local_llm(
            base,
            bridge=bridge,
            question=question,
        )


# ── Multi-Layer Trace ────────────────────────────────────────────────


class MultiLayerTrace:
    """Traces decisions across network layers."""

    def __init__(self) -> None:
        self._layers: Dict[str, List[DecisionNode]] = {}

    def add(self, node: DecisionNode) -> None:
        self._layers.setdefault(node.layer_id, []).append(node)

    @property
    def layer_ids(self) -> List[str]:
        return list(self._layers.keys())

    def spikes_at_layer(self, layer_id: str) -> int:
        return sum(1 for n in self._layers.get(layer_id, []) if n.decision == SpikeDecision.SPIKE)

    def spike_rate_at_layer(self, layer_id: str) -> float:
        nodes = self._layers.get(layer_id, [])
        if not nodes:
            return 0.0
        return sum(1 for n in nodes if n.decision == SpikeDecision.SPIKE) / len(nodes)

    def propagation_path(self) -> List[Dict[str, Any]]:
        """Per-layer spike rate for visualising propagation."""
        return [
            {"layer": lid, "spike_rate": self.spike_rate_at_layer(lid), "count": len(nodes)}
            for lid, nodes in self._layers.items()
        ]


# ── Symbolic Path ────────────────────────────────────────────────────


@dataclass
class SymbolicPathStep:
    """One step in a symbolic decision path."""

    neuron_id: str
    decision: SpikeDecision
    reason: str


class SymbolicPath:
    """Human-readable symbolic path: input → encoding → decision."""

    def __init__(self) -> None:
        self.steps: List[SymbolicPathStep] = []

    def add(self, neuron_id: str, decision: SpikeDecision, reason: str) -> None:
        self.steps.append(SymbolicPathStep(neuron_id, decision, reason))

    @property
    def length(self) -> int:
        return len(self.steps)

    def to_list(self) -> List[Dict[str, str]]:
        return [
            {"neuron": s.neuron_id, "decision": s.decision.value, "reason": s.reason}
            for s in self.steps
        ]


# ── Main Explainability Engine ───────────────────────────────────────


class ExplainabilityEngine:
    """End-to-end explainability: replay + decision tree + provenance."""

    def __init__(self, seed: int = 0xACE1) -> None:
        self.seed = seed
        self.replay = LFSRReplay(seed)
        self.tree = SpikeDecisionTree()
        self.provenance = ProvenanceTrace()
        self.temporal = TemporalWindow()
        self.multi_layer = MultiLayerTrace()
        self.symbolic = SymbolicPath()
        self._replayed_bitstreams: Dict[str, np.ndarray[Any, Any]] = {}

    def explain_spike(
        self,
        neuron_id: str,
        threshold_q16: int,
        bitstream_length: int,
        spike_threshold_count: int,
        scc: float = 0.0,
        timestep: int = 0,
        layer_id: str = "",
        contributing_neurons: Optional[List[str]] = None,
    ) -> DecisionNode:
        """Explain one spike decision via deterministic replay.

        Replays the LFSR from the current seed state, generates the
        exact bitstream, and records the decision in the tree and trace.
        """
        _validate_replay_parameters(
            threshold_q16=threshold_q16,
            bitstream_length=bitstream_length,
            spike_threshold_count=spike_threshold_count,
        )

        self.provenance.add_step(
            "input",
            f"Neuron {neuron_id}: threshold_q16={threshold_q16}, length={bitstream_length}",
        )

        replay = LFSRReplay(self.seed)
        bitstream = replay.encode(threshold_q16, bitstream_length)
        self._replayed_bitstreams[neuron_id] = bitstream

        self.provenance.add_step(
            "encoding",
            f"LFSR encode seed={self.seed:#06x}, threshold={threshold_q16}",
            data=bitstream,
        )

        node = self.tree.add_decision(
            neuron_id=neuron_id,
            bitstream=bitstream,
            threshold=spike_threshold_count,
            scc=scc,
            timestep=timestep,
            layer_id=layer_id,
            contributing_neurons=contributing_neurons,
            threshold_q16=threshold_q16,
        )

        self.temporal.add(node)
        self.multi_layer.add(node)

        reason = (
            f"popcount({node.popcount}) {'≥' if node.decision == SpikeDecision.SPIKE else '<'} "
            f"threshold({spike_threshold_count})"
        )
        self.symbolic.add(neuron_id, node.decision, reason)

        self.provenance.add_step(
            "decision",
            f"Neuron {neuron_id}: {node.decision.value} (popcount={node.popcount}/{spike_threshold_count})",
            metadata={
                "popcount": node.popcount,
                "threshold": spike_threshold_count,
                "probability": node.probability,
                "margin": node.margin.margin,
                "scc_influence": node.scc_influence,
            },
        )

        return node

    def verify(
        self,
        regulatory: Optional[RegulatoryMetadata] = None,
        formal_properties: Optional[List[FormalPropertyLink]] = None,
    ) -> VerifiabilityReport:
        """Generate a verifiability report with full replay check."""
        self.provenance.finalize()

        all_match = True
        for nid, stored_bs in self._replayed_bitstreams.items():
            fresh = LFSRReplay(self.seed)
            node = self.tree.get_node(nid)
            if node is not None:
                re_bs = fresh.encode(
                    threshold=node.threshold_q16,
                    length=node.bitstream_length,
                )
                if not np.array_equal(
                    stored_bs[: min(len(stored_bs), len(re_bs))],
                    re_bs[: min(len(stored_bs), len(re_bs))],
                ):
                    all_match = False

        return VerifiabilityReport(
            chain_hash=self.provenance.chain_hash,
            num_steps=self.provenance.num_steps,
            all_hashes_valid=all_match,
            decision_tree_depth=self.tree.depth,
            total_spikes=self.tree.num_spikes,
            replay_seed=self.seed,
            replay_matches=all_match,
            regulatory=regulatory,
            formal_properties=formal_properties or [],
        )

    def replay_bitstream(
        self,
        threshold_q16: int,
        length: int,
    ) -> np.ndarray[Any, Any]:
        """Replay a bitstream from the engine's seed (for external comparison)."""
        _validate_replay_parameters(
            threshold_q16=threshold_q16,
            bitstream_length=length,
            length_name="length",
        )
        replay = LFSRReplay(self.seed)
        return replay.encode(threshold_q16, length)

    def sensitivity(
        self,
        node: DecisionNode,
        perturbations: Optional[List[int]] = None,
    ) -> List[SensitivityResult]:
        """Run sensitivity analysis on a decision node."""
        return SensitivityAnalyzer.analyze(node, perturbations)

    def attribute(
        self,
        target: DecisionNode,
        input_bitstreams: Dict[str, np.ndarray[Any, Any]],
        weights: Optional[Dict[str, float]] = None,
    ) -> CausalAttribution:
        """Compute causal attribution for a decision."""
        return CausalAttributor.attribute(target, input_bitstreams, weights)

    def explain_spike_with_local_llm(
        self,
        neuron_id: str,
        threshold_q16: int,
        bitstream_length: int,
        spike_threshold_count: int,
        *,
        bridge: Any,
        scc: float = 0.0,
        timestep: int = 0,
        layer_id: str = "",
        contributing_neurons: Optional[List[str]] = None,
        question: str = (
            "Explain this spike decision for a human operator in two short paragraphs. "
            "Do not change or invent numeric values."
        ),
    ) -> tuple[DecisionNode, str]:
        """Run the deterministic explainability path, then enhance it locally."""
        node = self.explain_spike(
            neuron_id=neuron_id,
            threshold_q16=threshold_q16,
            bitstream_length=bitstream_length,
            spike_threshold_count=spike_threshold_count,
            scc=scc,
            timestep=timestep,
            layer_id=layer_id,
            contributing_neurons=contributing_neurons,
        )
        text = NaturalLanguageExplainer.explain_node_with_local_llm(
            node,
            bridge=bridge,
            question=question,
        )
        return node, text

# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Traceable Stochastic Explainability Layer

"""Bitstream-level explanations of threshold spike decisions, with LFSR replay and provenance.

A spike decision here is a popcount compared with a threshold count. The
module regenerates the bitstream of a decision from a 16-bit LFSR seed,
records the decision, a provenance trace of hashed steps and several views
of the recorded decisions (by layer, by timestep, as a symbolic path), and
derives margins, threshold sensitivity, a weighted-popcount attribution and
text explanations from them. The provenance trace's chain hash covers each
step's stage and data hash; the module itself never compares it with a
stored value.

Two behaviours shape what an explanation shows: every replay starts from the
engine's seed, so decisions with the same threshold and length get the same
bitstream whatever the neuron; and :class:`ExplainabilityEngine` records its
decisions without parent links, so its decision tree is a root with
unlinked nodes beside it.
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

    Parameters
    ----------
    seed : int
        Non-zero seed. The register keeps its low 16 bits, so a seed that is a
        non-zero multiple of 65536 starts a register of 0, which never leaves
        0.

    Attributes
    ----------
    initial_seed : int
        The seed as given, not reduced to 16 bits.
    reg : int
        The current 16-bit register value.

    Raises
    ------
    ValueError
        If ``seed`` is 0.
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
        """Generate a bitstream by comparing LFSR output against threshold.

        Each bit is 1 when the current register value is below the threshold,
        compared before the register steps, so the first bit compares the
        seed's register value.

        Parameters
        ----------
        threshold : int
            Comparison threshold in Q16 (0 to 65535 gives probabilities from 0
            to just below 1).
        length : int
            Number of bits.

        Returns
        -------
        numpy.ndarray
            The bits as uint8; the register has advanced ``length`` steps.
        """
        bits = np.zeros(length, dtype=np.uint8)
        for i in range(length):
            bits[i] = 1 if self.reg < threshold else 0
            self.step()
        return bits

    def reset(self) -> None:
        """Reset to initial seed for replay.

        The register is set to ``initial_seed`` without the 16-bit reduction
        the constructor applies, so for a seed above 65535 the register after
        a reset differs from the one after construction.
        """
        self.reg = self.initial_seed


# ── Decision Tree ────────────────────────────────────────────────────


class SpikeDecision(Enum):
    """Outcome of a threshold decision.

    :meth:`SpikeDecisionTree.add_decision` sets ``SPIKE`` or ``NO_SPIKE``;
    ``UNDETERMINED`` is the default of a node built by hand.
    """

    SPIKE = "spike"
    NO_SPIKE = "no_spike"
    UNDETERMINED = "undetermined"


@dataclass
class DecisionMargin:
    """How close a decision was to flipping.

    Attributes
    ----------
    popcount : int
        Ones in the bitstream.
    threshold : int
        Threshold count of the decision.
    margin : int
        popcount - threshold; negative for no spike.
    confidence : float
        |margin| / bitstream length; 0.0 for an empty bitstream.
    """

    popcount: int
    threshold: int
    margin: int  # popcount - threshold (negative = no spike)
    confidence: float  # |margin| / bitstream_length


@dataclass
class DecisionNode:
    """One node in a spike decision tree.

    Attributes
    ----------
    neuron_id : str
        Name of the deciding neuron.
    popcount : int
        Ones in the observed bitstream.
    threshold : int
        Threshold count; a popcount at or above it is a spike.
    bitstream_length : int
        Bits observed.
    probability : float
        popcount / bitstream_length.
    scc_context : float
        Stochastic cross-correlation given with the decision.
    scc_influence : float
        |scc_context| × probability.
    decision : SpikeDecision
        The outcome.
    children : list of DecisionNode
        Nodes recorded with this node as their parent.
    bitstream_hash : str
        First 16 hexadecimal characters of the SHA-256 of the bitstream's
        bytes.
    timestep : int
        Timestep of the decision.
    threshold_q16 : int
        Q16 threshold the bitstream was encoded with, 0 when not given.
    layer_id : str
        Layer of the neuron.
    contributing_neurons : list of str
        Names of the inputs given with the decision.
    """

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
        """Whether the node has no children."""
        return len(self.children) == 0

    @property
    def margin(self) -> DecisionMargin:
        """The decision's margin: popcount - threshold, and that over the bitstream length."""
        m = self.popcount - self.threshold
        conf = abs(m) / self.bitstream_length if self.bitstream_length > 0 else 0.0
        return DecisionMargin(self.popcount, self.threshold, m, conf)


class SpikeDecisionTree:
    """Captures "why this spike fired" as a verifiable decision tree.

    The first node recorded without a parent becomes the root; later nodes
    without a parent are kept in the tree's list (and counted) but linked to
    nothing, so the depth, the spike path and the dictionary form cover only
    the root and its descendants.

    Attributes
    ----------
    root : DecisionNode or None
        The first node recorded without a parent.
    """

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
        """Record a spike decision from bitstream observation.

        Parameters
        ----------
        neuron_id : str
            Name of the deciding neuron.
        bitstream : numpy.ndarray
            The observed bits.
        threshold : int
            Threshold count; a popcount at or above it is a spike.
        scc : float
            Stochastic cross-correlation of the inputs.
        parent : DecisionNode or None
            Node to attach this one to as a child.
        timestep : int
            Timestep of the decision.
        layer_id : str
            Layer of the neuron.
        contributing_neurons : list of str or None
            Names of the inputs.
        threshold_q16 : int
            Q16 threshold the bitstream was encoded with.

        Returns
        -------
        DecisionNode
            The recorded node: probability popcount / length (0.0 for an empty
            bitstream), SCC influence |scc| × probability, and the hash of the
            bitstream's bytes.
        """
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
        """Number of levels below and including the root; 0 without a root."""
        if self.root is None:
            return 0
        return self._compute_depth(self.root)

    def _compute_depth(self, node: DecisionNode) -> int:
        if not node.children:
            return 1
        return 1 + max(self._compute_depth(c) for c in node.children)

    @property
    def num_spikes(self) -> int:
        """Number of recorded nodes that spiked, unlinked ones included."""
        return sum(1 for n in self._nodes if n.decision == SpikeDecision.SPIKE)

    @property
    def num_nodes(self) -> int:
        """Number of recorded nodes, unlinked ones included."""
        return len(self._nodes)

    def nodes_at_layer(self, layer_id: str) -> List[DecisionNode]:
        """Return all nodes at a given layer."""
        return [n for n in self._nodes if n.layer_id == layer_id]

    def nodes_at_timestep(self, timestep: int) -> List[DecisionNode]:
        """Return all nodes at a given timestep."""
        return [n for n in self._nodes if n.timestep == timestep]

    def get_node(self, neuron_id: str) -> Optional[DecisionNode]:
        """Look up a node by neuron ID.

        Returns the first node recorded with that name, or ``None``.
        """
        for n in self._nodes:
            if n.neuron_id == neuron_id:
                return n
        return None

    def spike_path(self) -> List[DecisionNode]:
        """Return the chain of spiking nodes from root down.

        The spiking nodes of the root's subtree in depth-first order; a
        non-spiking node does not stop the walk below it.
        """
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
        """Return the root's subtree as nested dictionaries.

        Returns
        -------
        dict of str to Any
            Each node's name, popcount, threshold, probability, decision,
            hash, SCC influence, margin, confidence, timestep, layer, inputs
            and children; empty without a root.
        """
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
    """One step in a full provenance chain.

    Attributes
    ----------
    stage : str
        Name of the stage, for example ``"input"`` or ``"decision"``.
    description : str
        Text of the step.
    data_hash : str
        First 16 hexadecimal characters of the SHA-256 of the step's data, or
        of its description when it has no data.
    timestamp_ns : int
        ``time.perf_counter_ns()`` when the step was recorded: a monotonic
        counter, not a wall-clock time.
    metadata : dict of str to Any
        Extra values of the step.
    """

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
        """Record one provenance step.

        Parameters
        ----------
        stage : str
            Name of the stage.
        description : str
            Text of the step.
        data : numpy.ndarray or None
            Data whose bytes are hashed; without it the description is hashed.
        metadata : dict of str to Any or None
            Extra values of the step.

        Returns
        -------
        ProvenanceStep
            The recorded step. Steps can be added after :meth:`finalize`.
        """
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
        """Mark the trace complete; it does not stop further steps."""
        self._complete = True

    @property
    def is_complete(self) -> bool:
        """Whether :meth:`finalize` has been called."""
        return self._complete

    @property
    def num_steps(self) -> int:
        """Number of recorded steps."""
        return len(self._steps)

    @property
    def chain_hash(self) -> str:
        """Hash of the entire provenance chain for tamper detection.

        The SHA-256, as 64 hexadecimal characters, of every step's data hash
        and stage in order. Descriptions enter only through the data hash of
        a step without data; metadata and timestamps do not enter.
        """
        h = hashlib.sha256()
        for step in self._steps:
            h.update(step.data_hash.encode())
            h.update(step.stage.encode())
        return h.hexdigest()

    def to_list(self) -> List[Dict[str, Any]]:
        """Return the steps as dictionaries of their five fields, in order."""
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
    """IEC 62304 / FDA SaMD traceability fields.

    Free-text fields named after those traceability items; the module stores
    them in a report and checks none of them.

    Attributes
    ----------
    device_class, risk_level, intended_use, software_version : str
        Device classification, risk level, intended use and software version.
    udi : str
        Unique device identifier.
    sudi_hash : str
        Hash of the software and its configuration, as supplied.
    operator_id : str
        Operator of the run.
    review_status : str
        Review state, ``"pending"`` by default.
    """

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
    """Cross-reference to SymbiYosys formal verification.

    A record of a property's name, file and status as supplied; the module
    neither runs nor reads the proof.

    Attributes
    ----------
    property_name : str
        Name of the property.
    property_file : str
        File of the property.
    status : str
        ``"proven"``, ``"cex"`` or ``"unverified"`` (the default), as supplied.
    bounded_depth : int
        Proof depth, as supplied.
    engine : str
        Proof engine, ``"sby"`` by default.
    """

    property_name: str
    property_file: str = ""
    status: str = "unverified"  # proven / cex / unverified
    bounded_depth: int = 0
    engine: str = "sby"


@dataclass
class VerifiabilityReport:
    """Formal audit report with hash verification.

    Built by :meth:`ExplainabilityEngine.verify`.

    Attributes
    ----------
    chain_hash : str
        The provenance trace's chain hash.
    num_steps : int
        Steps of the provenance trace.
    all_hashes_valid : bool
        The replay comparison's result, the same value as ``replay_matches``;
        no hash is checked.
    decision_tree_depth : int
        Depth of the decision tree.
    total_spikes : int
        Recorded nodes that spiked.
    replay_seed : int
        The engine's seed.
    replay_matches : bool
        Whether every replayed bitstream matched its stored one.
    regulatory : RegulatoryMetadata or None
        Traceability fields, as supplied.
    formal_properties : list of FormalPropertyLink
        Property records, as supplied.
    """

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
    """Result of a 'what-if' threshold perturbation.

    Attributes
    ----------
    neuron_id : str
        Name of the neuron.
    original_threshold, perturbed_threshold : int
        Threshold count before and after the change.
    original_decision, perturbed_decision : SpikeDecision
        Decision before and after the change.
    flipped : bool
        Whether the two decisions differ.
    """

    neuron_id: str
    original_threshold: int
    perturbed_threshold: int
    original_decision: SpikeDecision
    perturbed_decision: SpikeDecision
    flipped: bool


class SensitivityAnalyzer:
    """Counterfactual analysis: would the decision flip if the threshold moved by ±N.

    Each perturbed decision compares the node's popcount with the shifted
    threshold count; the bitstream is not encoded again.
    """

    @staticmethod
    def analyze(
        node: DecisionNode,
        perturbations: Optional[List[int]] = None,
    ) -> List[SensitivityResult]:
        """Decide a node again with shifted threshold counts.

        Parameters
        ----------
        node : DecisionNode
            The decision to perturb.
        perturbations : list of int or None
            Threshold changes; ``None`` uses -10, -5, -1, 1, 5 and 10.

        Returns
        -------
        list of SensitivityResult
            One result per change, in order; a perturbed threshold below 0 is
            clamped to 0.
        """
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
        """Smallest threshold change that flips the decision.

        margin + 1 for a spike (raise the threshold above the popcount) and
        the negative margin itself for no spike (lower it to the popcount).
        """
        m = node.margin
        if m.margin >= 0:
            return m.margin + 1
        return m.margin


# ── Causal Attribution ───────────────────────────────────────────────


@dataclass
class CausalAttribution:
    """Attribution of a spike to upstream neurons.

    Attributes
    ----------
    target_neuron : str
        Name of the explained neuron.
    attributions : dict of str to float
        Weighted popcount of each input.
    total_contribution : float
        Sum of the attributions.
    """

    target_neuron: str
    attributions: Dict[str, float]  # source_neuron -> contribution weight
    total_contribution: float

    @property
    def top_contributors(self) -> List[Tuple[str, float]]:
        """Sorted (descending) list of contributing neurons."""
        return sorted(self.attributions.items(), key=lambda x: x[1], reverse=True)


class CausalAttributor:
    """Computes causal attribution from input neurons to output spike.

    The attribution of an input is its popcount times its weight; the target
    contributes only its name. No intervention or counterfactual is run.
    """

    @staticmethod
    def attribute(
        target: DecisionNode,
        input_bitstreams: Dict[str, np.ndarray[Any, Any]],
        weights: Optional[Dict[str, float]] = None,
    ) -> CausalAttribution:
        """Compute per-input-neuron contribution to the target popcount.

        Parameters
        ----------
        target : DecisionNode
            The explained decision; only its name is used.
        input_bitstreams : dict of str to numpy.ndarray
            Bitstream of each input neuron.
        weights : dict of str to float or None
            Weight of each input; a missing one, or no mapping, counts 1.0.

        Returns
        -------
        CausalAttribution
            Each input's sum of bits times its weight, and their total.
        """
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
    """One field that differs between two explanations.

    Attributes
    ----------
    field : str
        Name of the field.
    value_a, value_b : Any
        The field's value in the first and the second node.
    """

    field: str
    value_a: Any
    value_b: Any


class ExplanationDiff:
    """Compares two decision nodes to find divergence points."""

    @staticmethod
    def diff(a: DecisionNode, b: DecisionNode) -> List[DiffEntry]:
        """List the fields in which two nodes differ.

        Compared: neuron name, popcount, threshold, bitstream length,
        probability, SCC context, decision and bitstream hash. Timestep, layer,
        inputs, children and the Q16 threshold are not compared.

        Parameters
        ----------
        a, b : DecisionNode
            The two nodes.

        Returns
        -------
        list of DiffEntry
            One entry per differing field, in the order above.
        """
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
        """Record a node under its timestep."""
        self._windows.setdefault(node.timestep, []).append(node)

    def spike_rate_at(self, timestep: int) -> float:
        """Return the fraction of a timestep's nodes that spiked; 0.0 for an unseen timestep."""
        nodes = self._windows.get(timestep, [])
        if not nodes:
            return 0.0
        return sum(1 for n in nodes if n.decision == SpikeDecision.SPIKE) / len(nodes)

    def active_timesteps(self) -> List[int]:
        """Return the recorded timesteps in ascending order."""
        return sorted(self._windows.keys())

    def peak_timestep(self) -> int:
        """Timestep with the highest spike rate.

        On a tie the timestep recorded first wins; 0 when nothing is recorded.
        """
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
        """Number of distinct recorded timesteps."""
        return len(self._windows)


# ── Natural Language Explainer ───────────────────────────────────────


class NaturalLanguageExplainer:
    """Generates human-readable explanation strings."""

    @staticmethod
    def explain_node(node: DecisionNode) -> str:
        """Describe one decision in English.

        A spike reads "exceeded threshold … by N bits", also when the popcount
        equals the threshold (N is 0); no spike reads "fell short … by N bits".
        A positive SCC context and up to five input names are added; a
        negative SCC context is not mentioned.

        Parameters
        ----------
        node : DecisionNode
            The decision.

        Returns
        -------
        str
            The description.
        """
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
        """Describe an attribution by its three largest contributors and the total.

        Parameters
        ----------
        attr : CausalAttribution
            The attribution.

        Returns
        -------
        str
            The description.
        """
        top = attr.top_contributors[:3]
        parts = [f"{nid} ({w:.1f})" for nid, w in top]
        return (
            f"Spike at {attr.target_neuron} was primarily caused by: "
            f"{', '.join(parts)}. Total input contribution: {attr.total_contribution:.1f}."
        )

    @staticmethod
    def explain_sensitivity(results: List[SensitivityResult]) -> str:
        """Describe the smallest tested threshold change that flips the decision.

        Parameters
        ----------
        results : list of SensitivityResult
            Results of :meth:`SensitivityAnalyzer.analyze`.

        Returns
        -------
        str
            The flipping change of smallest size, or a sentence that no tested
            change flips the decision.
        """
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
        The module does not check the returned text against the numbers of
        the base explanation.

        Parameters
        ----------
        text : str
            The base explanation.
        bridge : Any
            An object whose ``chat(prompt)`` returns a response with ``text``.
        question : str
            Instruction placed before the base explanation.

        Returns
        -------
        str
            The response's text.
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
        """Generate a local-LLM-enhanced explanation for one decision node.

        Parameters
        ----------
        node : DecisionNode
            The decision; :meth:`explain_node` gives the base explanation.
        bridge : Any
            As for :meth:`enhance_with_local_llm`.
        question : str
            Instruction placed before the base explanation.

        Returns
        -------
        str
            The response's text.
        """
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
        """Record a node under its layer."""
        self._layers.setdefault(node.layer_id, []).append(node)

    @property
    def layer_ids(self) -> List[str]:
        """The recorded layers in the order first seen."""
        return list(self._layers.keys())

    def spikes_at_layer(self, layer_id: str) -> int:
        """Return how many of a layer's nodes spiked; 0 for an unseen layer."""
        return sum(1 for n in self._layers.get(layer_id, []) if n.decision == SpikeDecision.SPIKE)

    def spike_rate_at_layer(self, layer_id: str) -> float:
        """Return the fraction of a layer's nodes that spiked; 0.0 for an unseen layer."""
        nodes = self._layers.get(layer_id, [])
        if not nodes:
            return 0.0
        return sum(1 for n in nodes if n.decision == SpikeDecision.SPIKE) / len(nodes)

    def propagation_path(self) -> List[Dict[str, Any]]:
        """Per-layer spike rate for visualising propagation.

        One dictionary per layer in the order first seen, with the layer, its
        spike rate and its node count.
        """
        return [
            {"layer": lid, "spike_rate": self.spike_rate_at_layer(lid), "count": len(nodes)}
            for lid, nodes in self._layers.items()
        ]


# ── Symbolic Path ────────────────────────────────────────────────────


@dataclass
class SymbolicPathStep:
    """One step in a symbolic decision path.

    Attributes
    ----------
    neuron_id : str
        Name of the neuron.
    decision : SpikeDecision
        Its decision.
    reason : str
        Text of the reason.
    """

    neuron_id: str
    decision: SpikeDecision
    reason: str


class SymbolicPath:
    """Human-readable symbolic path: input → encoding → decision.

    Attributes
    ----------
    steps : list of SymbolicPathStep
        The steps in the order added.
    """

    def __init__(self) -> None:
        self.steps: List[SymbolicPathStep] = []

    def add(self, neuron_id: str, decision: SpikeDecision, reason: str) -> None:
        """Append one step.

        Parameters
        ----------
        neuron_id : str
            Name of the neuron.
        decision : SpikeDecision
            Its decision.
        reason : str
            Text of the reason.
        """
        self.steps.append(SymbolicPathStep(neuron_id, decision, reason))

    @property
    def length(self) -> int:
        """Number of steps in the path."""
        return len(self.steps)

    def to_list(self) -> List[Dict[str, str]]:
        """Return the steps as dictionaries of neuron, decision value and reason."""
        return [
            {"neuron": s.neuron_id, "decision": s.decision.value, "reason": s.reason}
            for s in self.steps
        ]


# ── Main Explainability Engine ───────────────────────────────────────


class ExplainabilityEngine:
    """End-to-end explainability: replay + decision tree + provenance.

    Parameters
    ----------
    seed : int
        Non-zero LFSR seed every replay starts from.

    Attributes
    ----------
    seed : int
        The LFSR seed.
    replay : LFSRReplay
        A replay engine built from the seed; the engine's methods build their
        own and do not use it.
    tree : SpikeDecisionTree
        Every explained decision, recorded without a parent.
    provenance : ProvenanceTrace
        Three steps per explained decision: input, encoding and decision.
    temporal : TemporalWindow
        The decisions by timestep.
    multi_layer : MultiLayerTrace
        The decisions by layer.
    symbolic : SymbolicPath
        One step per decision with its popcount comparison.

    Raises
    ------
    ValueError
        If ``seed`` is 0.
    """

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

        Replays the LFSR from the engine's seed, generates the bitstream, and
        records the decision in the tree and trace. Every call starts from
        the seed, so two neurons with the same Q16 threshold and length get
        the same bitstream. The bitstream is stored under the neuron's name,
        replacing one stored before.

        Parameters
        ----------
        neuron_id : str
            Name of the neuron.
        threshold_q16 : int
            Encoding threshold in Q16, from 0 to 65535.
        bitstream_length : int
            Number of bits; must be positive.
        spike_threshold_count : int
            Threshold count of the decision, from 0 to ``bitstream_length``.
        scc : float
            Stochastic cross-correlation of the inputs.
        timestep : int
            Timestep of the decision.
        layer_id : str
            Layer of the neuron.
        contributing_neurons : list of str or None
            Names of the inputs.

        Returns
        -------
        DecisionNode
            The recorded decision.

        Raises
        ------
        ValueError
            If a parameter is outside its range; nothing is recorded then.
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
        """Generate a verifiability report with full replay check.

        Marks the provenance trace complete, then replays every stored
        bitstream from the seed with the Q16 threshold and length of the
        first node recorded under that neuron's name and compares the common
        prefix. A neuron explained twice with other parameters is reported as
        a mismatch only when the two bitstreams differ in their common
        prefix; a changed length alone, or a nearby threshold, still matches.

        Parameters
        ----------
        regulatory : RegulatoryMetadata or None
            Traceability fields to carry in the report.
        formal_properties : list of FormalPropertyLink or None
            Property records to carry in the report.

        Returns
        -------
        VerifiabilityReport
            The chain hash, step count, tree depth, spike count, seed and the
            replay result, given as both ``replay_matches`` and
            ``all_hashes_valid``.
        """
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
        """Replay a bitstream from the engine's seed (for external comparison).

        Parameters
        ----------
        threshold_q16 : int
            Encoding threshold in Q16, from 0 to 65535.
        length : int
            Number of bits; must be positive.

        Returns
        -------
        numpy.ndarray
            The bits as uint8; nothing is recorded.

        Raises
        ------
        ValueError
            If a parameter is outside its range.
        """
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
        """Run sensitivity analysis on a decision node.

        Parameters
        ----------
        node : DecisionNode
            The decision to perturb.
        perturbations : list of int or None
            Threshold changes, as for :meth:`SensitivityAnalyzer.analyze`.

        Returns
        -------
        list of SensitivityResult
            One result per change.
        """
        return SensitivityAnalyzer.analyze(node, perturbations)

    def attribute(
        self,
        target: DecisionNode,
        input_bitstreams: Dict[str, np.ndarray[Any, Any]],
        weights: Optional[Dict[str, float]] = None,
    ) -> CausalAttribution:
        """Compute causal attribution for a decision.

        Parameters
        ----------
        target : DecisionNode
            The explained decision.
        input_bitstreams : dict of str to numpy.ndarray
            Bitstream of each input neuron.
        weights : dict of str to float or None
            Weight of each input.

        Returns
        -------
        CausalAttribution
            As :meth:`CausalAttributor.attribute` computes it.
        """
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
        """Run the deterministic explainability path, then enhance it locally.

        Parameters
        ----------
        neuron_id : str
            Name of the neuron.
        threshold_q16 : int
            Encoding threshold in Q16, from 0 to 65535.
        bitstream_length : int
            Number of bits; must be positive.
        spike_threshold_count : int
            Threshold count of the decision, from 0 to ``bitstream_length``.
        bridge : Any
            As for :meth:`NaturalLanguageExplainer.enhance_with_local_llm`.
        scc : float
            Stochastic cross-correlation of the inputs.
        timestep : int
            Timestep of the decision.
        layer_id : str
            Layer of the neuron.
        contributing_neurons : list of str or None
            Names of the inputs.
        question : str
            Instruction placed before the base explanation.

        Returns
        -------
        tuple of DecisionNode and str
            The recorded decision and the bridge's text.
        """
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

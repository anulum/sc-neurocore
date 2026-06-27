# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — GOTM Self-Learning Brain

"""GOTM self-learning brain module.

Composes the quantum cognition layer (SpinPoolMPS, HybridFisherPosnerLIF,
FisherPosnerQuantumBridge) with a local LLM and the GOTM content indexer
to create a self-learning system that internalises the entire God of the
Math collection.

The brain operates in a learning loop:

    1. Content indexer walks GOTM repositories → text chunks
    2. Chunks are embedded into numerical vectors
    3. Optional local LLM provides semantic guidance (FOCUS / EXPLORE / STABILIZE)
    4. Guidance sets quantum bridge target coherence
    5. Numerical vectors drive neuron input currents
    6. Spikes update the non-local entanglement map
    7. Identity persists via ArcaneNeuron ``v_deep`` compartment

    The LLM path is opt-in and strictly local (llama-server) — no external API
    calls.  Content is read-only.  Learning affects only the spin pool and
    neuron weights, never the source repositories.

Example
-------
::

    brain = GOTMBrain(n_neurons=32)
    brain.learn_from_repo("/path/to/SC-NEUROCORE")
    state = brain.get_learning_state()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from .bridge_adapter import FisherPosnerQuantumBridge, compute_max_qubits
from .content_indexer import ContentChunk, embed_chunks, index_gotm_repo
from .fisher_posner import HybridFisherPosnerLIF
from .spin_pool import SpinPoolMPS

logger = logging.getLogger(__name__)

# Attempt to import the local LLM adapter
try:
    import sys as _sys

    # The agentic-shared llm module is available on the GOTM workstation
    _sys.path.insert(0, "/media/anulum/724AA8E84AA8AA75/agentic-shared")
    from llm import Endpoint as _LLMEndpoint
    from llm import chat as _llm_chat

    HAS_LLM = True
except (ImportError, ModuleNotFoundError):
    _llm_chat = None
    _LLMEndpoint = None
    HAS_LLM = False


# Directive → target coherence mapping
_DIRECTIVE_COHERENCE = {
    "FOCUS": 0.8,
    "EXPLORE": 0.4,
    "STABILIZE": 0.6,
}

# Directive → learning rate mapping
_DIRECTIVE_LR = {
    "FOCUS": 0.2,
    "EXPLORE": 0.05,
    "STABILIZE": 0.1,
}


@dataclass
class LearningStep:
    """Record of a single learning step for telemetry."""

    step_index: int
    directive: str
    target_coherence: float
    n_spikes: int
    avg_atp: float
    avg_entanglement: float
    chunk_summary: str
    chunk_sha256: str

    def to_dict(self) -> dict[str, Any]:
        """Serialise to JSON-compatible dict."""
        return {
            "step": self.step_index,
            "directive": self.directive,
            "target_coherence": self.target_coherence,
            "n_spikes": self.n_spikes,
            "avg_atp": self.avg_atp,
            "avg_entanglement": self.avg_entanglement,
            "chunk_summary": self.chunk_summary,
            "chunk_sha256": self.chunk_sha256,
        }


class GOTMBrain:
    """Self-learning brain for the God of the Math collection.

    Composes quantum cognition classes with a local LLM to create a
    neural system that learns from GOTM content.

    Parameters
    ----------
    n_neurons : int
        Number of neurons (also determines spin pool sites and qubit count).
    bridge_backend : str
        Quantum bridge backend (``"emulated"``, ``"pennylane"``,
        ``"ibm_aer"``, ``"ibm_qiskit"``, or ``"auto"``). The default is
        explicit emulation so repository learning never silently switches to
        an expensive simulator because an optional package is installed.
    seed : int or None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        n_neurons: int = 32,
        bridge_backend: str = "emulated",
        seed: int | None = 42,
        llm_endpoint: Any = None,
    ) -> None:
        if n_neurons < 1:
            raise ValueError(f"n_neurons must be >= 1, got {n_neurons}")

        self.n_neurons = n_neurons
        self.pool = SpinPoolMPS(n_sites=n_neurons)

        # RAM-aware qubit auto-sizing: never exceed available memory
        max_qubits = compute_max_qubits()
        n_qubits = min(n_neurons, max_qubits)
        logger.info(
            "GOTMBrain: %d neurons, %d qubits (RAM limit: %d)",
            n_neurons,
            n_qubits,
            max_qubits,
        )
        self.bridge = FisherPosnerQuantumBridge(
            n_qubits=n_qubits,
            backend=bridge_backend,
        )
        self.neurons = [HybridFisherPosnerLIF(i, self.pool) for i in range(n_neurons)]
        self._rng = np.random.default_rng(seed)
        self._history: list[LearningStep] = []
        self._total_steps = 0
        # LLM endpoint for fleet routing (agentic-shared Endpoint object)
        self._llm_endpoint = llm_endpoint

    def get_llm_guidance(self, context_summary: str) -> str:
        """Query the configured local LLM for a learning directive.

        Parameters
        ----------
        context_summary : str
            Text summary of the current content being processed.

        Returns
        -------
        str
            One of ``"FOCUS"``, ``"EXPLORE"``, or ``"STABILIZE"``.
            Falls back to ``"STABILIZE"`` if no local LLM endpoint was
            explicitly configured.
        """
        if self._llm_endpoint is None or not HAS_LLM or _llm_chat is None:
            logger.debug("LLM disabled or unavailable, using fallback directive: STABILIZE")
            return "STABILIZE"

        prompt = (
            f"Analyse this mathematical context from the God of Math collection:\n"
            f"{context_summary[:500]}\n\n"
            f"Provide a neuromodulation directive for the Quantum Neurocore.\n"
            f"Choose exactly one: FOCUS, EXPLORE, or STABILIZE.\n"
            f"Reply with the word only."
        )

        try:
            kwargs: dict[str, Any] = {
                "system": "You are a Mathematical Architect.",
                "temperature": 0.2,
                "max_tokens": 10,
            }
            kwargs["endpoint"] = self._llm_endpoint
            response = _llm_chat(prompt, **kwargs)
            directive: str = response.strip().split()[0].upper().strip(".,:;")
            if directive in _DIRECTIVE_COHERENCE:
                return directive
        except Exception as exc:
            logger.warning("LLM error: %s", exc)

        return "STABILIZE"

    def process_content(
        self,
        input_vector: np.ndarray[Any, Any],
        directive: str,
    ) -> list[int]:
        """Process a content vector through the neural network.

        Parameters
        ----------
        input_vector : np.ndarray[Any, Any]
            Numerical vector from content embedding, shape ``(n_neurons,)``
            or broadcastable.
        directive : str
            LLM directive (``"FOCUS"``, ``"EXPLORE"``, ``"STABILIZE"``).

        Returns
        -------
        list[int]
            Indices of neurons that spiked.
        """
        target = _DIRECTIVE_COHERENCE.get(directive, 0.6)
        lr = _DIRECTIVE_LR.get(directive, 0.1)

        # Optimise quantum bridge phases (if PennyLane available)
        self.bridge.optimize_phases(target_coherence=target, learning_rate=lr, n_steps=1)

        # Prepare input currents
        if len(input_vector) < self.n_neurons:
            padded = np.zeros(self.n_neurons, dtype=np.float64)
            padded[: len(input_vector)] = input_vector
            input_vector = padded
        elif len(input_vector) > self.n_neurons:
            input_vector = input_vector[: self.n_neurons]

        # Scale to biologically plausible current range
        currents = input_vector * 60.0

        # Step all neurons
        spikes: list[int] = []
        for i, neuron in enumerate(self.neurons):
            _, spiked = neuron.step(float(currents[i]))
            if spiked:
                spikes.append(i)

        return spikes

    def learn_step(self, chunk: ContentChunk, vector: np.ndarray[Any, Any]) -> LearningStep:
        """Execute a single learning step on one content chunk.

        Parameters
        ----------
        chunk : ContentChunk
            The content chunk being processed.
        vector : np.ndarray[Any, Any]
            Embedded numerical vector for the chunk.

        Returns
        -------
        LearningStep
            Record of the learning step.
        """
        directive = self.get_llm_guidance(chunk.summary)
        spikes = self.process_content(vector, directive)

        avg_atp = float(np.mean([n.atp_level for n in self.neurons]))
        avg_ent = float(np.mean(self.pool.entanglement_map))

        step_record = LearningStep(
            step_index=self._total_steps,
            directive=directive,
            target_coherence=_DIRECTIVE_COHERENCE.get(directive, 0.6),
            n_spikes=len(spikes),
            avg_atp=avg_atp,
            avg_entanglement=avg_ent,
            chunk_summary=chunk.summary,
            chunk_sha256=chunk.sha256,
        )
        self._history.append(step_record)
        self._total_steps += 1

        logger.info(
            "Step %d [%s]: %d spikes, ATP=%.3f, ent=%.4f — %s",
            step_record.step_index,
            directive,
            len(spikes),
            avg_atp,
            avg_ent,
            chunk.summary[:60],
        )
        return step_record

    def learn_from_repo(
        self,
        repo_path: str,
        repo_name: str | None = None,
        max_chunks: int | None = None,
    ) -> list[LearningStep]:
        """Index and learn from an entire GOTM repository.

        Parameters
        ----------
        repo_path : str
            Path to the repository root.
        repo_name : str, optional
            Override repository name.
        max_chunks : int, optional
            Limit the number of chunks to process.

        Returns
        -------
        list[LearningStep]
            Records of all learning steps.
        """
        chunks = index_gotm_repo(repo_path, repo_name)
        if max_chunks is not None:
            chunks = chunks[:max_chunks]

        vectors = embed_chunks(chunks, n_dims=self.n_neurons)
        steps: list[LearningStep] = []

        for chunk, vector in zip(chunks, vectors):
            step = self.learn_step(chunk, vector)
            steps.append(step)

        logger.info(
            "Learned from %s: %d chunks, %d total spikes",
            repo_name or repo_path,
            len(steps),
            sum(s.n_spikes for s in steps),
        )
        return steps

    def get_learning_state(self) -> dict[str, Any]:
        """Return full learning state for inspection and persistence."""
        return {
            "n_neurons": self.n_neurons,
            "total_steps": self._total_steps,
            "total_spikes": sum(n._total_spikes for n in self.neurons),
            "total_metabolic_failures": sum(n._metabolic_failures for n in self.neurons),
            "avg_atp": float(np.mean([n.atp_level for n in self.neurons])),
            "avg_entanglement": float(np.mean(self.pool.entanglement_map)),
            "pool_state": self.pool.get_state(),
            "bridge_backend": self.bridge.backend,
            "has_llm": bool(HAS_LLM and self._llm_endpoint is not None),
            "history_length": len(self._history),
        }

    def get_history(self) -> list[dict[str, Any]]:
        """Return learning history as a list of dicts."""
        return [s.to_dict() for s in self._history]

    def save_state(self, path: str) -> None:
        """Persist full brain state (v_deep) to a JSON file.

        Serialises neuron states, spin pool entanglement map, learning
        history, and step counter so that a learning session can be
        resumed later.

        Parameters
        ----------
        path : str
            Output file path (JSON).
        """
        import json

        state = {
            "n_neurons": self.n_neurons,
            "total_steps": self._total_steps,
            "pool_state": self.pool.get_state(),
            "neuron_states": [n.get_state() for n in self.neurons],
            "bridge_backend": self.bridge.backend,
            "history": self.get_history(),
        }
        with open(path, "w") as f:
            # Convert numpy types for JSON compatibility
            json.dump(
                state,
                f,
                indent=2,
                default=lambda o: float(o) if hasattr(o, "__float__") else str(o),
            )
        logger.info("Brain state saved to %s (%d steps)", path, self._total_steps)

    def load_state(self, path: str) -> None:
        """Restore brain state (v_deep) from a previously saved JSON file.

        Parameters
        ----------
        path : str
            Input file path (JSON, from ``save_state``).
        """
        import json

        with open(path) as f:
            state = json.load(f)

        # Validate compatibility
        if state["n_neurons"] != self.n_neurons:
            raise ValueError(f"State has {state['n_neurons']} neurons, brain has {self.n_neurons}")

        # Restore pool
        pool_state = state["pool_state"]
        self.pool.entanglement_map = np.array(pool_state["entanglement_map"])
        self.pool._measurement_count = pool_state.get("measurement_count", 0)

        # Restore neurons
        for neuron, nstate in zip(self.neurons, state["neuron_states"]):
            neuron.Vm = nstate.get("Vm", -70.0)
            neuron.atp_level = nstate.get("atp_level", 1.0)
            neuron._total_spikes = nstate.get("total_spikes", 0)
            neuron._metabolic_failures = nstate.get("metabolic_failures", 0)

        # Restore history
        self._total_steps = state["total_steps"]
        self._history = [
            LearningStep(
                step_index=h["step"],
                directive=h["directive"],
                target_coherence=h["target_coherence"],
                n_spikes=h["n_spikes"],
                avg_atp=h["avg_atp"],
                avg_entanglement=h["avg_entanglement"],
                chunk_summary=h.get("chunk_summary", ""),
                chunk_sha256=h.get("chunk_sha256", ""),
            )
            for h in state.get("history", [])
        ]

        logger.info("Brain state loaded from %s (%d steps)", path, self._total_steps)

    def reset(self) -> None:
        """Reset all neurons, spin pool, and history."""
        self.pool.reset()
        for neuron in self.neurons:
            neuron.reset_state()
        self._history.clear()
        self._total_steps = 0

    def __repr__(self) -> str:
        """Return a concise representation of the brain learning state."""
        return (
            f"GOTMBrain(n_neurons={self.n_neurons}, "
            f"steps={self._total_steps}, "
            f"backend={self.bridge.backend!r})"
        )


__all__ = ["GOTMBrain", "LearningStep", "HAS_LLM"]

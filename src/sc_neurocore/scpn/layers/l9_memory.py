# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SCPN L9 Holographic Memory Layer (Stochastic Implementation)

"""SCPN L9: holographic memory layer (stochastic implementation).

Hopfield-style associative memory with TSVF-inspired forward/backward
overlap for memory retrieval quality.

E = -1/2 * sum_ij w_ij s_i s_j  (Hopfield energy)
Retrieval = <Phi|Psi> / <Phi|Phi>  (TSVF weak-value proxy)

Ref: Paper 9 — Memory Imprint-Existential Holograph.
"""

from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class L9_StochasticParameters:
    """Stochastic configuration parameters for the SCPN holographic memory layer."""

    n_memory_slots: int = 64
    bitstream_length: int = 1024
    retrieval_gain: float = 0.8
    imprint_rate: float = 0.3
    decay_rate: float = 0.02
    phase_field_coupling: float = 0.1  # from L8
    boundary_cue_coupling: float = 1.0
    rng_seed: Optional[int] = None


class L9_MemoryLayer:
    """Hopfield associative memory with stochastic bitstream encoding."""

    def __init__(self, params: Optional[L9_StochasticParameters] = None):
        self.params = params or L9_StochasticParameters()
        self._validate_params(self.params)
        n = self.params.n_memory_slots
        self.patterns = np.zeros((n, n))  # weight matrix
        self._rng = np.random.default_rng(self.params.rng_seed)
        self.state = self._rng.choice([-1.0, 1.0], size=n).astype(np.float64)
        self.n_stored = 0
        self.time = 0.0

    def store(self, pattern: np.ndarray[Any, Any]) -> None:
        """Hebbian imprint: W += pattern ⊗ pattern."""
        p = self._pattern_vector(pattern)
        self.patterns += self.params.imprint_rate * np.outer(p, p) / self.params.n_memory_slots
        np.fill_diagonal(self.patterns, 0)
        self.n_stored += 1

    def step(
        self,
        dt: float,
        l8_input: Optional[Dict[str, Any]] = None,
        boundary_cue: Optional[np.ndarray[Any, Any]] = None,
        ebs_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Advance the holographic memory layer one timestep and return its output state."""
        if not math.isfinite(float(dt)) or float(dt) <= 0.0:
            raise ValueError("dt must be finite and positive")
        cue = self._boundary_cue_vector(boundary_cue)
        boundary_context = self._boundary_context(ebs_context)
        self.time += dt
        n = self.params.n_memory_slots

        # Hopfield dynamics: async update (random subset)
        update_mask = self._rng.random(n) < 0.3
        h = self.params.retrieval_gain * (self.patterns @ self.state)
        if l8_input is not None:
            h += self.params.phase_field_coupling * self._l8_phase_reference_drive(l8_input)
        proposed_state = np.where(h > 0.0, 1.0, np.where(h < 0.0, -1.0, self.state))
        self.state = np.where(update_mask, proposed_state, self.state)

        qec_syndrome = np.zeros(n, dtype=np.uint8)
        recovery_operator = np.zeros(n, dtype=np.float64)
        if cue is not None:
            cue_sign = np.sign(cue)
            cue_sign = np.where(cue_sign == 0.0, 1.0, cue_sign)
            mismatch = np.sign(self.state) != cue_sign
            qec_syndrome = mismatch.astype(np.uint8)
            recovery_mask = mismatch & (self._rng.random(n) < self.params.boundary_cue_coupling)
            recovery_operator = recovery_mask.astype(np.float64)
            self.state = np.where(recovery_mask, cue_sign, self.state)

        # Retrieval quality: overlap with stored patterns
        activation = (self.state + 1) / 2  # map [-1,1] -> [0,1]
        activation = np.clip(activation, 0, 1)

        # Decay
        self.patterns *= math.exp(-self.params.decay_rate * float(dt))

        rands = self._rng.random((n, self.params.bitstream_length))
        output_bitstreams = (rands < activation[:, None]).astype(np.uint8)

        energy = -0.5 * float(self.state @ self.patterns @ self.state)
        holographic_entropy = self._holographic_entropy(activation)
        memory_free_energy = float(max(0.0, -energy) + holographic_entropy)

        return {
            "state": self.state.copy(),
            "energy": energy,
            "retrieval_quality": self._retrieval_quality(),
            "holographic_entropy": holographic_entropy,
            "memory_free_energy": memory_free_energy,
            "qec_syndrome": qec_syndrome,
            "recovery_operator": recovery_operator,
            "boundary_context_id": boundary_context["ebs_id"],
            "boundary_terminals": boundary_context["terminal_set"],
            "output_bitstreams": output_bitstreams,
        }

    def _retrieval_quality(self) -> float:
        if self.n_stored == 0:
            return 0.0
        h = self.params.retrieval_gain * (self.patterns @ self.state)
        return float(np.mean(np.sign(h) == np.sign(self.state)))

    def get_global_metric(self) -> float:
        """Return the scalar global metric summarising this layer's state."""
        return self._retrieval_quality()

    @staticmethod
    def _validate_params(params: L9_StochasticParameters) -> None:
        if not isinstance(params.n_memory_slots, int) or isinstance(params.n_memory_slots, bool):
            raise ValueError("n_memory_slots must be a positive integer")
        if params.n_memory_slots <= 0:
            raise ValueError("n_memory_slots must be positive")
        if not isinstance(params.bitstream_length, int) or isinstance(
            params.bitstream_length, bool
        ):
            raise ValueError("bitstream_length must be a positive integer")
        if params.bitstream_length <= 0:
            raise ValueError("bitstream_length must be positive")
        if not math.isfinite(float(params.retrieval_gain)) or params.retrieval_gain < 0.0:
            raise ValueError("retrieval_gain must be finite and non-negative")
        if not math.isfinite(float(params.imprint_rate)) or not 0.0 <= params.imprint_rate <= 1.0:
            raise ValueError("imprint_rate must be finite and in [0, 1]")
        if not math.isfinite(float(params.decay_rate)) or params.decay_rate < 0.0:
            raise ValueError("decay_rate must be finite and non-negative")
        if (
            not math.isfinite(float(params.phase_field_coupling))
            or params.phase_field_coupling < 0.0
        ):
            raise ValueError("phase_field_coupling must be finite and non-negative")
        if (
            not math.isfinite(float(params.boundary_cue_coupling))
            or params.boundary_cue_coupling < 0.0
            or params.boundary_cue_coupling > 1.0
        ):
            raise ValueError("boundary_cue_coupling must be finite and in [0, 1]")
        if params.rng_seed is not None:
            if isinstance(params.rng_seed, bool) or not isinstance(params.rng_seed, int):
                raise ValueError("rng_seed must be a non-negative integer or None")
            if params.rng_seed < 0:
                raise ValueError("rng_seed must be a non-negative integer or None")

    def _pattern_vector(self, pattern: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        values = np.asarray(pattern, dtype=np.float64).reshape(-1)
        if values.size < self.params.n_memory_slots:
            raise ValueError("pattern must contain at least n_memory_slots values")
        trimmed = values[: self.params.n_memory_slots]
        if not np.all(np.isfinite(trimmed)):
            raise ValueError("pattern must contain only finite values")
        return np.sign(trimmed)

    @staticmethod
    def _cosmic_alignment(value: Any) -> float:
        values = np.asarray(value, dtype=np.float64)
        if values.shape != ():
            raise ValueError("cosmic_alignment must be a finite scalar")
        cosmic_alignment = float(values)
        if not math.isfinite(cosmic_alignment):
            raise ValueError("cosmic_alignment must be a finite scalar")
        return cosmic_alignment

    @classmethod
    def _l8_phase_reference_drive(cls, l8_input: Dict[str, Any]) -> float:
        if "memory_imprint_drive" in l8_input:
            return cls._memory_imprint_drive(l8_input["memory_imprint_drive"])
        if "cosmic_alignment" in l8_input:
            return cls._cosmic_alignment(l8_input["cosmic_alignment"])
        return 0.0

    @staticmethod
    def _memory_imprint_drive(payload: Any) -> float:
        if not isinstance(payload, dict):
            raise ValueError("memory_imprint_drive must be a mapping")
        try:
            amplitude_values = np.asarray(payload["reference_amplitude"], dtype=np.float64)
            phase_values = np.asarray(payload["reference_phase"], dtype=np.float64)
        except KeyError as exc:
            raise ValueError(
                "memory_imprint_drive requires reference_amplitude and reference_phase"
            ) from exc
        if amplitude_values.shape != () or phase_values.shape != ():
            raise ValueError("memory_imprint_drive fields must be finite scalars")
        amplitude = float(amplitude_values)
        phase = float(phase_values)
        if not math.isfinite(amplitude) or not 0.0 <= amplitude <= 1.0:
            raise ValueError(
                "memory_imprint_drive reference_amplitude must be finite and in [0, 1]"
            )
        if not math.isfinite(phase):
            raise ValueError("memory_imprint_drive reference_phase must be finite")
        drive = float(amplitude * math.cos(phase))
        if "reference_real" in payload:
            reference_real_values = np.asarray(payload["reference_real"], dtype=np.float64)
            if reference_real_values.shape != ():
                raise ValueError("memory_imprint_drive reference_real must be a finite scalar")
            reference_real = float(reference_real_values)
            if not math.isfinite(reference_real) or not math.isclose(
                reference_real, drive, rel_tol=1e-9, abs_tol=1e-12
            ):
                raise ValueError("memory_imprint_drive reference_real is inconsistent")
        return drive

    def _boundary_cue_vector(
        self, boundary_cue: Optional[np.ndarray[Any, Any]]
    ) -> Optional[np.ndarray[Any, Any]]:
        if boundary_cue is None:
            return None
        values = np.asarray(boundary_cue, dtype=np.float64).reshape(-1)
        if values.size != self.params.n_memory_slots:
            raise ValueError("boundary_cue must contain exactly n_memory_slots values")
        if not np.all(np.isfinite(values)):
            raise ValueError("boundary_cue must contain only finite values")
        if np.any(values < -1.0) or np.any(values > 1.0):
            raise ValueError("boundary_cue values must be within [-1, 1]")
        return values

    @staticmethod
    def _boundary_context(ebs_context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if ebs_context is None:
            return {"ebs_id": None, "terminal_set": ()}
        if "ebs_id" not in ebs_context:
            raise ValueError("ebs_context must include ebs_id")
        terminals = tuple(ebs_context.get("terminal_set", ()))
        valid_terminals = {"T1", "T2", "T3", "T4", "T5", "T6", "T7"}
        if not terminals or any(terminal not in valid_terminals for terminal in terminals):
            raise ValueError("terminal_set must contain valid T1-T7 terminal identifiers")
        return {"ebs_id": str(ebs_context["ebs_id"]), "terminal_set": terminals}

    @staticmethod
    def _holographic_entropy(activation: np.ndarray[Any, Any]) -> float:
        probs = np.clip(np.asarray(activation, dtype=np.float64), 1e-12, 1.0 - 1e-12)
        entropy = -(probs * np.log2(probs) + (1.0 - probs) * np.log2(1.0 - probs))
        return float(np.mean(entropy))

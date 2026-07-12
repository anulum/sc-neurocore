# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — DNA sequence design

"""Deterministic constrained sequence design for molecular circuits."""

from __future__ import annotations

import hashlib
from typing import Tuple

import numpy as np

from .dna_types import (
    _GC_TARGET_HIGH,
    _GC_TARGET_LOW,
    _MAX_HOMOPOLYMER,
    _RECOGNITION_LENGTH,
    _TOEHOLD_LENGTH,
)


class SequenceDesigner:
    """Deterministic DNA sequence generator with constraint satisfaction.

    Generates sequences that satisfy GC content, homopolymer, and
    orthogonality constraints using a seed-based deterministic algorithm.
    This ensures reproducible designs without requiring NUPACK.

    Parameters
    ----------
    seed : int
        Random seed for reproducible sequence generation.
    gc_target : tuple[float, float]
        Acceptable GC content range (default 0.40–0.60).
    max_homopolymer : int
        Maximum consecutive identical nucleotides (default 3).
    """

    def __init__(
        self,
        seed: int = 42,
        gc_target: Tuple[float, float] = (_GC_TARGET_LOW, _GC_TARGET_HIGH),
        max_homopolymer: int = _MAX_HOMOPOLYMER,
    ) -> None:
        self._rng = np.random.default_rng(seed)
        self._gc_target = gc_target
        self._max_homopolymer = max_homopolymer
        self._used_sequences: list[str] = []

    def generate(self, length: int, name: str = "seq") -> str:
        """Generate a sequence satisfying all constraints.

        Uses rejection sampling with guided nucleotide selection to
        maintain GC content within bounds while avoiding homopolymer
        runs.

        Parameters
        ----------
        length : int
            Desired sequence length.
        name : str
            Identifier for debugging (used in hash seed).

        Returns
        -------
        str
            Valid nucleotide sequence (A, C, G, T).
        """
        nucs = ["A", "C", "G", "T"]
        best_seq = ""
        best_score = float("inf")

        seed_hash = int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)
        rng = np.random.default_rng(self._rng.integers(0, 2**31) + seed_hash)

        for _attempt in range(200):
            seq: list[str] = []
            gc_count = 0

            for i in range(length):
                # Determine allowed nucleotides
                allowed = list(nucs)

                # Prevent homopolymer runs
                if len(seq) >= self._max_homopolymer:
                    last_n = seq[-self._max_homopolymer :]
                    if len(set(last_n)) == 1:
                        allowed = [n for n in allowed if n != last_n[0]]

                # Bias toward GC target
                if i > 0:
                    current_gc = gc_count / i
                    if current_gc < self._gc_target[0]:
                        # Need more GC
                        weights = [0.15, 0.35, 0.35, 0.15]
                    elif current_gc > self._gc_target[1]:
                        # Need more AT
                        weights = [0.35, 0.15, 0.15, 0.35]
                    else:
                        weights = [0.25, 0.25, 0.25, 0.25]
                    # Zero out disallowed
                    weights = [w if nucs[j] in allowed else 0.0 for j, w in enumerate(weights)]
                else:
                    weights = [1.0 if n in allowed else 0.0 for n in nucs]

                total = sum(weights)
                probs = [w / total for w in weights]

                nuc = rng.choice(nucs, p=probs)
                seq.append(nuc)
                if nuc in "GC":
                    gc_count += 1

            candidate = "".join(seq)
            gc = gc_count / length
            score = abs(gc - 0.5) * 10

            # Penalize homopolymer runs
            max_run = 1
            cur_run = 1
            for i in range(1, len(candidate)):
                if candidate[i] == candidate[i - 1]:
                    cur_run += 1
                    max_run = max(max_run, cur_run)
                else:
                    cur_run = 1
            if max_run > self._max_homopolymer:
                score += (max_run - self._max_homopolymer) * 5

            # Penalize similarity to existing sequences
            for existing in self._used_sequences:
                overlap = sum(1 for a, b in zip(candidate, existing) if a == b)
                similarity = overlap / max(len(candidate), len(existing), 1)
                if similarity > 0.7:
                    score += similarity * 10

            if score < best_score:
                best_score = score
                best_seq = candidate

            if score < 0.5:
                break

        self._used_sequences.append(best_seq)
        return best_seq

    def generate_complement(self, sequence: str) -> str:
        """Return the Watson-Crick complement (3' → 5')."""
        table = str.maketrans("ACGT", "TGCA")
        return sequence.translate(table)[::-1]

    def generate_toehold(self, name: str = "toehold") -> str:
        """Generate a toehold domain (6 nt)."""
        return self.generate(_TOEHOLD_LENGTH, name)

    def generate_recognition(self, name: str = "recog") -> str:
        """Generate a recognition domain (15 nt)."""
        return self.generate(_RECOGNITION_LENGTH, name)

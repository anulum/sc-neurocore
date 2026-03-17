# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Reasoning trace to spike pattern encoding

"""Encode reasoning traces as temporal spike patterns.

Uses locality-sensitive hashing to map text chunks to neuron groups,
then generates Poisson spike trains weighted by chunk salience.
Pure numpy, no external NLP dependencies.
"""

from __future__ import annotations

import re

import numpy as np


def _tokenize(text: str) -> list[str]:
    """Split text into sentence-like chunks on punctuation boundaries."""
    chunks = re.split(r'[.!?;]\s+', text.strip())
    return [c.strip() for c in chunks if c.strip()]


def _word_set(chunk: str) -> set[str]:
    """Extract lowercased alphanumeric tokens."""
    return set(re.findall(r'[a-z0-9]+', chunk.lower()))


def _salience(chunk: str, position: int, total: int) -> float:
    """Salience score based on position and lexical density.

    First and last sentences score higher (inverted-U weighting).
    Longer chunks with more unique words score higher.
    """
    pos_weight = 1.0
    if position == 0 or position == total - 1:
        pos_weight = 1.5
    words = _word_set(chunk)
    density = min(len(words) / 20.0, 1.0)
    return pos_weight * (0.3 + 0.7 * density)


class TraceEncoder:
    """Encode reasoning traces as temporal spike patterns.

    LSH maps text chunks to neuron groups. Poisson spike trains
    weighted by chunk salience deliver the encoded signal.
    """

    def __init__(self, n_neurons=500, hash_dims=64, seed=42):
        self.n_neurons = n_neurons
        self.hash_dims = hash_dims
        self.seed = seed
        rng = np.random.default_rng(seed)
        self._projection = rng.standard_normal((hash_dims, n_neurons))

    def _hash_to_neurons(self, text: str) -> np.ndarray:
        """Map text to an activation vector over neurons via LSH."""
        words = _word_set(text)
        if not words:
            return np.zeros(self.n_neurons, dtype=np.float64)

        word_vec = np.zeros(self.hash_dims, dtype=np.float64)
        for w in words:
            h = hash(w) % self.hash_dims
            word_vec[h] += 1.0

        if word_vec.sum() > 0:
            word_vec /= word_vec.sum()

        activations = word_vec @ self._projection
        activations = np.clip(activations, 0, None)
        total = activations.sum()
        if total > 0:
            activations /= total
        return activations

    def encode(self, text: str, duration_ms=100, dt=0.001) -> np.ndarray:
        """Convert text to spike pattern array (n_neurons, n_steps).

        Each chunk maps to a neuron group; Poisson spike trains
        are generated with rates proportional to chunk salience.
        """
        chunks = _tokenize(text)
        if not chunks:
            chunks = [text] if text.strip() else [""]

        n_steps = int(duration_ms / (dt * 1000))
        spikes = np.zeros((self.n_neurons, n_steps), dtype=np.float64)
        rng = np.random.default_rng(self.seed + 1)

        steps_per_chunk = max(1, n_steps // len(chunks))

        for idx, chunk in enumerate(chunks):
            activations = self._hash_to_neurons(chunk)
            weight = _salience(chunk, idx, len(chunks))
            rates = activations * weight * 100.0  # Hz base rate

            t_start = idx * steps_per_chunk
            t_end = min(t_start + steps_per_chunk, n_steps)

            for t in range(t_start, t_end):
                p_spike = rates * dt
                spikes[:, t] = (rng.random(self.n_neurons) < p_spike).astype(np.float64)

        return spikes

    def encode_key_value(self, key: str, value: str) -> np.ndarray:
        """Encode a key-value pair as a combined spike pattern."""
        combined = f"{key}: {value}"
        return self.encode(combined, duration_ms=150)

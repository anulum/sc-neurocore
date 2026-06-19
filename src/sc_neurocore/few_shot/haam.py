# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Spike-based few-shot meta-learning

"""Hebbian-Augmented Associative Memory for few-shot SNN learning.

Learn from 1-5 examples using spike-timing plasticity, not gradients.
Store support examples as spike patterns, retrieve via cosine similarity.

Reference: HAAM (BICS 2024)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


class HebbianFewShot:
    """Hebbian few-shot learner using associative memory.

    Support examples stored via one-shot Hebbian weight update.
    Query classified by comparing spike-rate representation to
    stored prototypes.

    Parameters
    ----------
    n_features : int
        Input feature dimension.
    n_classes : int
        Number of classes to support.
    lr_hebbian : float
        Hebbian learning rate for support storage.
    """

    def __init__(self, n_features: int, n_classes: int, lr_hebbian: float = 0.1):
        self.n_features = n_features
        self.n_classes = n_classes
        self.lr_hebbian = lr_hebbian
        # Associative memory: one weight vector per class
        self.memory = np.zeros((n_classes, n_features))
        self._counts = np.zeros(n_classes, dtype=int)

    def store(self, spike_pattern: np.ndarray[Any, Any], label: int) -> None:
        """Store one support example via Hebbian update.

        Parameters
        ----------
        spike_pattern : ndarray of shape (n_features,) or (T, n_features)
            Spike pattern or spike rate vector.
        label : int
            Class label.
        """
        if spike_pattern.ndim > 1:
            pattern = spike_pattern.mean(axis=0)
        else:
            pattern = spike_pattern.astype(np.float64)

        # Hebbian update: strengthen connections for this class
        self.memory[label] += self.lr_hebbian * pattern
        self._counts[label] += 1

    def query(self, spike_pattern: np.ndarray[Any, Any]) -> int:
        """Classify a query pattern by cosine similarity to stored memories.

        Parameters
        ----------
        spike_pattern : ndarray of shape (n_features,) or (T, n_features)

        Returns
        -------
        int — predicted class
        """
        if spike_pattern.ndim > 1:
            pattern = spike_pattern.mean(axis=0)
        else:
            pattern = spike_pattern.astype(np.float64)

        similarities = np.zeros(self.n_classes)
        for c in range(self.n_classes):
            if self._counts[c] == 0:
                continue
            mem_norm = np.linalg.norm(self.memory[c])
            pat_norm = np.linalg.norm(pattern)
            if mem_norm > 1e-10 and pat_norm > 1e-10:
                similarities[c] = np.dot(self.memory[c], pattern) / (mem_norm * pat_norm)

        return int(np.argmax(similarities))

    def few_shot_episode(
        self,
        support_x: list[np.ndarray[Any, Any]],
        support_y: list[int],
        query_x: list[np.ndarray[Any, Any]],
    ) -> list[int]:
        """Run a complete few-shot episode.

        Parameters
        ----------
        support_x : list of ndarray
            Support set spike patterns.
        support_y : list of int
            Support set labels.
        query_x : list of ndarray
            Query set spike patterns.

        Returns
        -------
        list of int — predicted labels for query set
        """
        self.reset()
        for pattern, label in zip(support_x, support_y):
            self.store(pattern, label)
        return [self.query(q) for q in query_x]

    def reset(self) -> None:
        """Clear the associative memory and per-slot usage counts."""
        self.memory[:] = 0
        self._counts[:] = 0


@dataclass
class SpikePrototypeNet:
    """Prototypical network in spike domain.

    Compute class prototypes as mean spike-rate vectors from support set.
    Classify queries by nearest prototype (Euclidean or cosine).

    Parameters
    ----------
    n_features : int
    metric : str
        'cosine' or 'euclidean'
    """

    n_features: int
    metric: str = "cosine"

    def classify(
        self,
        support_x: list[np.ndarray[Any, Any]],
        support_y: list[int],
        query_x: list[np.ndarray[Any, Any]],
    ) -> list[int]:
        """Classify query set using support set prototypes.

        Parameters
        ----------
        support_x : list of ndarray, shape (n_features,) or (T, n_features)
        support_y : list of int
        query_x : list of ndarray

        Returns
        -------
        list of int
        """
        # Compute prototypes
        classes = sorted(set(support_y))
        prototypes = {}
        for c in classes:
            patterns = [
                s.mean(axis=0) if s.ndim > 1 else s.astype(np.float64)
                for s, y in zip(support_x, support_y)
                if y == c
            ]
            prototypes[c] = np.mean(patterns, axis=0)

        # Classify queries
        predictions = []
        for q in query_x:
            qv = q.mean(axis=0) if q.ndim > 1 else q.astype(np.float64)
            best_c = classes[0]
            best_score = -float("inf")
            for c, proto in prototypes.items():
                if self.metric == "cosine":
                    n1, n2 = np.linalg.norm(qv), np.linalg.norm(proto)
                    score = np.dot(qv, proto) / max(n1 * n2, 1e-10)
                else:
                    score = -np.linalg.norm(qv - proto)
                if score > best_score:
                    best_score = score
                    best_c = c
            predictions.append(best_c)

        return predictions

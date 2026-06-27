# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Spike-based few-shot meta-learning

"""Spike-domain few-shot learners for associative-memory episodes.

The module provides two small deterministic learners for N-way K-shot spike
classification. ``HebbianFewShot`` stores support examples in class-indexed
associative memory and scores queries by cosine similarity. ``SpikePrototypeNet``
keeps no training state between calls; it computes support-set prototypes and
classifies query vectors by cosine similarity, Euclidean distance, or binary
Hamming distance.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]
Metric = Literal["cosine", "euclidean", "hamming"]


def _validate_positive_int(value: int, name: str) -> int:
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _as_feature_vector(pattern: ArrayLike, n_features: int, *, name: str) -> FloatArray:
    arr = np.asarray(pattern, dtype=np.float64)
    if arr.ndim == 1:
        vector = arr
    elif arr.ndim == 2:
        vector = arr.mean(axis=0)
    else:
        raise ValueError(f"{name} must have shape ({n_features},) or (T, {n_features})")

    if vector.shape != (n_features,):
        raise ValueError(f"{name} must resolve to {n_features} features")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values")
    return vector.astype(np.float64, copy=True)


def _validate_label(label: int, n_classes: int) -> int:
    if not isinstance(label, int) or label < 0 or label >= n_classes:
        raise ValueError(f"label must be an integer in [0, {n_classes})")
    return label


def _cosine_score(lhs: FloatArray, rhs: FloatArray) -> float:
    denom = float(np.linalg.norm(lhs) * np.linalg.norm(rhs))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(lhs, rhs) / denom)


def _metric_score(metric: Metric, query: FloatArray, prototype: FloatArray) -> float:
    if metric == "cosine":
        return _cosine_score(query, prototype)
    if metric == "euclidean":
        return -float(np.linalg.norm(query - prototype))

    query_bits = query > 0.0
    prototype_bits = prototype > 0.0
    return -float(np.mean(np.not_equal(query_bits, prototype_bits)))


class HebbianFewShot:
    """Class-indexed Hebbian memory for few-shot spike episodes.

    Parameters
    ----------
    n_features : int
        Number of spike-rate features per pattern after temporal averaging.
    n_classes : int
        Number of class slots stored by the associative memory.
    lr_hebbian : float, default=0.1
        Non-negative multiplier applied when support patterns are accumulated
        into their class memory rows.
    """

    def __init__(self, n_features: int, n_classes: int, lr_hebbian: float = 0.1) -> None:
        self.n_features = _validate_positive_int(n_features, "n_features")
        self.n_classes = _validate_positive_int(n_classes, "n_classes")
        self.lr_hebbian = float(lr_hebbian)
        if not np.isfinite(self.lr_hebbian) or self.lr_hebbian < 0.0:
            raise ValueError("lr_hebbian must be finite and non-negative")

        self.memory: FloatArray = np.zeros((self.n_classes, self.n_features), dtype=np.float64)
        self._counts: NDArray[np.int64] = np.zeros(self.n_classes, dtype=np.int64)

    def store(self, spike_pattern: ArrayLike, label: int) -> None:
        """Store one support pattern in the class memory.

        Parameters
        ----------
        spike_pattern : array_like
            Spike-rate vector with shape ``(n_features,)`` or spike train with
            shape ``(T, n_features)``. Temporal spike trains are averaged over
            the first axis before storage.
        label : int
            Class slot to update.

        Raises
        ------
        ValueError
            If the label is out of range or the pattern cannot be resolved to a
            finite feature vector.
        """
        class_index = _validate_label(label, self.n_classes)
        pattern = _as_feature_vector(spike_pattern, self.n_features, name="spike_pattern")
        self.memory[class_index] += self.lr_hebbian * pattern
        self._counts[class_index] += 1

    def query_scores(self, spike_pattern: ArrayLike) -> FloatArray:
        """Return cosine scores for a query against every stored class.

        Parameters
        ----------
        spike_pattern : array_like
            Query spike-rate vector or temporal spike train.

        Returns
        -------
        numpy.ndarray
            One score per class. Classes with no support examples receive a
            score of ``0``.
        """
        pattern = _as_feature_vector(spike_pattern, self.n_features, name="spike_pattern")
        scores = np.zeros(self.n_classes, dtype=np.float64)
        for class_index in range(self.n_classes):
            if self._counts[class_index] > 0:
                scores[class_index] = _cosine_score(self.memory[class_index], pattern)
        return scores

    def query(self, spike_pattern: ArrayLike) -> int:
        """Classify one query pattern by nearest stored memory.

        Parameters
        ----------
        spike_pattern : array_like
            Query spike-rate vector or temporal spike train.

        Returns
        -------
        int
            Predicted class label.

        Raises
        ------
        ValueError
            If no support examples have been stored.
        """
        if not np.any(self._counts):
            raise ValueError("at least one support example must be stored before query")
        return int(np.argmax(self.query_scores(spike_pattern)))

    def few_shot_episode(
        self,
        support_x: Sequence[ArrayLike],
        support_y: Sequence[int],
        query_x: Sequence[ArrayLike],
    ) -> list[int]:
        """Run one reset-store-query few-shot episode.

        Parameters
        ----------
        support_x : list of array_like
            Support spike patterns.
        support_y : list of int
            Class label for each support pattern.
        query_x : list of array_like
            Query spike patterns to classify after support storage.

        Returns
        -------
        list of int
            Predicted labels for the query set.

        Raises
        ------
        ValueError
            If the support pattern and label lists have different lengths.
        """
        if len(support_x) != len(support_y):
            raise ValueError("support_x and support_y must have the same length")

        self.reset()
        for pattern, label in zip(support_x, support_y, strict=True):
            self.store(pattern, label)
        return [self.query(query) for query in query_x]

    def export_weights(self) -> FloatArray:
        """Return a defensive copy of the class memory matrix.

        Returns
        -------
        numpy.ndarray
            Matrix with shape ``(n_classes, n_features)`` containing the
            accumulated Hebbian support memory.
        """
        return self.memory.copy()

    def reset(self) -> None:
        """Clear the memory matrix and support counts."""
        self.memory.fill(0.0)
        self._counts.fill(0)


@dataclass
class SpikePrototypeNet:
    """Nearest-prototype classifier for spike-rate few-shot episodes.

    Parameters
    ----------
    n_features : int
        Number of features per vector after temporal averaging.
    metric : {"cosine", "euclidean", "hamming"}, default="cosine"
        Distance or similarity metric used to score queries against support-set
        prototypes. ``hamming`` thresholds vectors at zero and scores by negative
        normalised bit disagreement.
    """

    n_features: int
    metric: Metric = "cosine"
    prototypes: dict[int, FloatArray] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        """Validate the prototype classifier configuration after dataclass init."""
        self.n_features = _validate_positive_int(self.n_features, "n_features")
        if self.metric not in {"cosine", "euclidean", "hamming"}:
            raise ValueError("metric must be one of: cosine, euclidean, hamming")

    def classify(
        self,
        support_x: Sequence[ArrayLike],
        support_y: Sequence[int],
        query_x: Sequence[ArrayLike],
    ) -> list[int]:
        """Classify query patterns from support-set prototypes.

        Parameters
        ----------
        support_x : list of array_like
            Support spike patterns with one-dimensional or temporal shapes.
        support_y : list of int
            Label for each support pattern.
        query_x : list of array_like
            Query spike patterns to classify.

        Returns
        -------
        list of int
            Predicted class labels.

        Raises
        ------
        ValueError
            If support inputs are empty, labels are mismatched, or any pattern
            cannot be resolved to a finite feature vector.
        """
        self.prototypes = self._build_prototypes(support_x, support_y)
        classes = sorted(self.prototypes)
        predictions: list[int] = []

        for query in query_x:
            query_vector = _as_feature_vector(query, self.n_features, name="query")
            best_class = classes[0]
            best_score = -float("inf")
            for class_index in classes:
                score = _metric_score(self.metric, query_vector, self.prototypes[class_index])
                if score > best_score:
                    best_score = score
                    best_class = class_index
            predictions.append(best_class)

        return predictions

    def export_prototypes(self) -> dict[int, FloatArray]:
        """Return defensive copies of the most recently computed prototypes.

        Returns
        -------
        dict of int to numpy.ndarray
            Mapping from class label to prototype vector.
        """
        return {label: prototype.copy() for label, prototype in self.prototypes.items()}

    def _build_prototypes(
        self,
        support_x: Sequence[ArrayLike],
        support_y: Sequence[int],
    ) -> dict[int, FloatArray]:
        if not support_x:
            raise ValueError("support_x must contain at least one support pattern")
        if len(support_x) != len(support_y):
            raise ValueError("support_x and support_y must have the same length")

        grouped: dict[int, list[FloatArray]] = {}
        for pattern, label in zip(support_x, support_y, strict=True):
            if not isinstance(label, int):
                raise ValueError("support_y labels must be integers")
            grouped.setdefault(label, []).append(
                _as_feature_vector(pattern, self.n_features, name="support pattern")
            )

        prototypes: dict[int, FloatArray] = {}
        for label, patterns in sorted(grouped.items()):
            prototype = np.mean(np.stack(patterns, axis=0), axis=0, dtype=np.float64)
            prototypes[label] = np.asarray(prototype, dtype=np.float64)
        return prototypes

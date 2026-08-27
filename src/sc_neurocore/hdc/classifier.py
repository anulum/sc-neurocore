# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — HDC centroid classifier with mistake-driven retraining

"""Centroid hypervector classifier with mistake-driven retraining.

Each class keeps a bipolar accumulator (the sum of ``2 * v - 1`` over
its training hypervectors); the class centroid is the accumulator's
sign, with exact zeros resolved by the owning encoder's bundle tie
policy. Prediction is the nearest centroid by Hamming distance, and
retraining applies the standard mistake-driven update: a misclassified
example is added to its true class and subtracted from the predicted
class, sharpening both centroids without touching correct examples.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from sc_neurocore.hdc.base import HDCEncoder


@dataclass
class CentroidHDClassifier:
    """Nearest-centroid classifier over binary hypervectors.

    Deterministic for a seeded encoder: fitting, prediction, and
    retraining consume randomness only through the encoder (and only
    when its tie policy is ``"random"``).
    """

    encoder: HDCEncoder
    _accumulators: dict[str, np.ndarray[Any, Any]] = field(
        init=False, repr=False, default_factory=dict
    )
    _centroids: dict[str, np.ndarray[Any, Any]] = field(
        init=False, repr=False, default_factory=dict
    )

    def _validate_vector(self, vector: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        candidate = np.asarray(vector)
        if candidate.shape != (self.encoder.dim,):
            raise ValueError(f"vector must have shape ({self.encoder.dim},)")
        if not bool(np.isin(candidate, (0, 1)).all()):
            raise ValueError("vector entries must be binary (0 or 1)")
        return candidate.astype(np.int64)

    def _bipolar(self, vector: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        bipolar: np.ndarray[Any, Any] = 2 * vector - 1
        return bipolar

    def _refresh_centroid(self, label: str) -> None:
        accumulator = self._accumulators[label]
        # The accumulator's sign is the majority of its bundled examples;
        # shifting by the accumulator range reuses the encoder's shared
        # majority kernel so exact zeros follow the same tie policy.
        magnitude = int(np.abs(accumulator).max(initial=1))
        count = 2 * magnitude
        self._centroids[label] = self.encoder.majority(accumulator + magnitude, count)

    def fit(self, vectors: list[np.ndarray[Any, Any]], labels: list[str]) -> None:
        """Accumulate the labelled hypervectors into their class centroids."""
        if not vectors or len(vectors) != len(labels):
            raise ValueError("vectors and labels must be non-empty and of equal length")
        for vector, label in zip(vectors, labels):
            validated = self._validate_vector(vector)
            if label not in self._accumulators:
                self._accumulators[label] = np.zeros(self.encoder.dim, dtype=np.int64)
            self._accumulators[label] += self._bipolar(validated)
        for label in set(labels):
            self._refresh_centroid(label)

    def predict(self, vector: np.ndarray[Any, Any]) -> str:
        """Return the label of the nearest centroid by Hamming distance."""
        if not self._centroids:
            raise ValueError("classifier has no fitted classes")
        validated = self._validate_vector(vector).astype(np.uint8)
        best_label = ""
        best_distance = self.encoder.dim + 1
        for label in sorted(self._centroids):
            distance = int(np.count_nonzero(np.bitwise_xor(validated, self._centroids[label])))
            if distance < best_distance:
                best_distance = distance
                best_label = label
        return best_label

    def retrain(
        self,
        vectors: list[np.ndarray[Any, Any]],
        labels: list[str],
        epochs: int = 1,
    ) -> list[int]:
        """Run mistake-driven retraining; return misclassifications per epoch."""
        if not isinstance(epochs, int) or isinstance(epochs, bool) or epochs < 1:
            raise ValueError("epochs must be a positive integer")
        if not vectors or len(vectors) != len(labels):
            raise ValueError("vectors and labels must be non-empty and of equal length")
        unknown = sorted(set(labels) - set(self._accumulators))
        if unknown:
            raise ValueError(f"retrain labels must be fitted classes; unknown: {unknown}")
        mistakes_per_epoch: list[int] = []
        for _ in range(epochs):
            mistakes = 0
            for vector, label in zip(vectors, labels):
                validated = self._validate_vector(vector)
                predicted = self.predict(validated.astype(np.uint8))
                if predicted == label:
                    continue
                mistakes += 1
                bipolar = self._bipolar(validated)
                self._accumulators[label] += bipolar
                self._accumulators[predicted] -= bipolar
                self._refresh_centroid(label)
                self._refresh_centroid(predicted)
            mistakes_per_epoch.append(mistakes)
        return mistakes_per_epoch

    def centroid(self, label: str) -> np.ndarray[Any, Any]:
        """Return a copy of one fitted class centroid."""
        if label not in self._centroids:
            raise ValueError(f"unknown class {label!r}")
        return self._centroids[label].copy()

    @property
    def classes(self) -> tuple[str, ...]:
        """Return the fitted class labels in sorted order."""
        return tuple(sorted(self._centroids))

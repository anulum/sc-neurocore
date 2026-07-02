# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Data ingestion and normalization for multimodal SC

"""Validated multimodal data ingestion for stochastic-computing pipelines."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

Array = np.ndarray[Any, Any]
DEFAULT_LABEL_KEY = "labels"


def _ensure_sample_axis(name: str, values: Array) -> int:
    """Return the sample count after rejecting scalar or empty arrays."""
    if values.ndim == 0:
        raise ValueError(f"modality {name!r} must have at least one sample axis")
    if values.shape[0] == 0:
        raise ValueError(f"modality {name!r} must contain at least one sample")
    return int(values.shape[0])


def _normalize_modality(name: str, values: Any) -> Array:
    """Return a finite float array normalized across its observed range."""
    array = np.asarray(values, dtype=float)
    _ensure_sample_axis(name, array)
    if not bool(np.all(np.isfinite(array))):
        raise ValueError(f"modality {name!r} must contain only finite values")

    arr_min = float(np.min(array))
    arr_max = float(np.max(array))
    if arr_max > arr_min:
        return (array - arr_min) / (arr_max - arr_min)
    return np.zeros_like(array, dtype=float)


def _validate_dataset_shapes(data: Mapping[str, Array], labels: Array) -> None:
    """Validate shared modality and label sample-axis lengths."""
    sample_count: int | None = None
    for name, values in data.items():
        current_count = _ensure_sample_axis(name, values)
        if sample_count is None:
            sample_count = current_count
        elif current_count != sample_count:
            raise ValueError("all modalities must share the same sample count")

    if sample_count is None:
        raise ValueError("dataset requires at least one modality")
    if labels.ndim == 0 or int(labels.shape[0]) != sample_count:
        raise ValueError("labels length must match the modality sample count")


@dataclass
class MultimodalDataset:
    """Validated multimodal training dataset.

    Parameters
    ----------
    data:
        Mapping from modality names to normalized arrays. The first axis is the
        sample axis and must have the same length for every modality.
    labels:
        Label array whose first axis matches the modality sample count.
    """

    data: dict[str, Array]
    labels: Array

    def __post_init__(self) -> None:
        """Validate dataset shape invariants after construction."""
        _validate_dataset_shapes(self.data, self.labels)

    def get_sample(self, idx: int) -> dict[str, Array]:
        """Return the per-modality arrays for one sample index."""
        return {k: v[idx] for k, v in self.data.items()}


class DataIngestor:
    """Normalize raw multimodal arrays into a `MultimodalDataset`.

    Parameters
    ----------
    label_key:
        Reserved key used to extract labels from the raw input mapping.
    """

    def __init__(self, label_key: str = DEFAULT_LABEL_KEY) -> None:
        """Initialize the ingestor with the reserved label key."""
        if not label_key:
            raise ValueError("label_key must be a non-empty string")
        self.label_key = label_key

    def prepare_dataset(self, raw_data: Mapping[str, Any]) -> MultimodalDataset:
        """Normalize and package raw multimodal data."""
        processed_data: dict[str, Array] = {}
        raw_labels = raw_data.get(self.label_key)

        for name, values in raw_data.items():
            if name == self.label_key:
                continue
            processed_data[name] = _normalize_modality(name, values)

        if not processed_data:
            raise ValueError("raw_data must contain at least one modality")

        first_array = next(iter(processed_data.values()))
        sample_count = int(first_array.shape[0])
        labels: Array = np.zeros(sample_count, dtype=int)
        if raw_labels is not None:
            labels = np.asarray(raw_labels)

        return MultimodalDataset(data=processed_data, labels=labels)

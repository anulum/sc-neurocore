# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for pipeline ingestion contracts

"""Contracts for multimodal dataset preparation and sample access."""

from __future__ import annotations

import numpy as np
import pytest

from sc_neurocore.pipeline.ingestion import DataIngestor, MultimodalDataset


def test_data_ingestor_preserves_labels_and_excludes_label_modality() -> None:
    raw = {
        "signals": np.arange(20, dtype=float).reshape(10, 2),
        "labels": np.array([0, 1] * 5),
    }

    dataset = DataIngestor().prepare_dataset(raw)

    assert isinstance(dataset, MultimodalDataset)
    assert set(dataset.data) == {"signals"}
    assert np.array_equal(dataset.labels, raw["labels"])
    assert dataset.data["signals"].dtype.kind == "f"
    assert float(dataset.data["signals"].min()) == 0.0
    assert float(dataset.data["signals"].max()) == 1.0


def test_multimodal_dataset_returns_named_sample_mapping() -> None:
    dataset = MultimodalDataset(
        data={"eeg": np.random.randn(10, 4), "emg": np.random.randn(10, 2)},
        labels=np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1]),
    )

    sample = dataset.get_sample(0)

    assert isinstance(sample, dict)
    assert {"eeg", "emg"} <= set(sample)


def test_data_ingestor_zeros_a_constant_modality() -> None:
    # A modality with no dynamic range cannot be min-max normalised; it must
    # collapse to zeros rather than divide by a zero range.
    dataset = DataIngestor().prepare_dataset(
        {"flat": [5.0, 5.0, 5.0], "varied": [0.0, 1.0, 2.0]}
    )

    assert np.array_equal(dataset.data["flat"], np.zeros(3))
    assert dataset.data["varied"].max() == 1.0 and dataset.data["varied"].min() == 0.0


def test_data_ingestor_rejects_empty_payload() -> None:
    with pytest.raises(ValueError, match="at least one modality"):
        DataIngestor().prepare_dataset({"labels": np.array([1, 0])})


def test_multimodal_dataset_rejects_empty_data_mapping() -> None:
    with pytest.raises(ValueError, match="at least one modality"):
        MultimodalDataset(data={}, labels=np.array([1, 0]))


def test_data_ingestor_rejects_empty_modalities() -> None:
    with pytest.raises(ValueError, match="at least one sample"):
        DataIngestor().prepare_dataset({"eeg": np.array([])})


def test_data_ingestor_rejects_mismatched_modality_lengths() -> None:
    with pytest.raises(ValueError, match="same sample count"):
        DataIngestor().prepare_dataset({"eeg": np.zeros((3, 2)), "emg": np.zeros((4, 2))})


def test_data_ingestor_rejects_label_length_mismatch() -> None:
    with pytest.raises(ValueError, match="labels length"):
        DataIngestor().prepare_dataset({"eeg": np.zeros((3, 2)), "labels": np.array([0, 1])})


def test_data_ingestor_rejects_non_finite_modality_values() -> None:
    with pytest.raises(ValueError, match="finite values"):
        DataIngestor().prepare_dataset({"eeg": np.array([0.0, np.nan, 1.0])})


def test_data_ingestor_rejects_scalar_modalities() -> None:
    with pytest.raises(ValueError, match="at least one sample axis"):
        DataIngestor().prepare_dataset({"eeg": 1.0})


def test_data_ingestor_rejects_empty_label_key() -> None:
    with pytest.raises(ValueError, match="label_key"):
        DataIngestor(label_key="")

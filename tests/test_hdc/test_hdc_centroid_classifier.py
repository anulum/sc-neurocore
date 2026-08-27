# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — HDC centroid classifier and retraining

"""Centroid fitting, nearest-centroid prediction, and retraining."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest

from sc_neurocore.hdc import CentroidHDClassifier, HDCEncoder

_DIM = 512


def _noisy_copies(
    rng: np.random.Generator,
    prototype: np.ndarray[Any, Any],
    count: int,
    flip_fraction: float,
) -> list[np.ndarray[Any, Any]]:
    copies = []
    flips = int(prototype.shape[0] * flip_fraction)
    for _ in range(count):
        vector = prototype.copy()
        positions = rng.choice(prototype.shape[0], size=flips, replace=False)
        vector[positions] ^= 1
        copies.append(vector)
    return copies


def _two_cluster_problem(
    flip_fraction: float,
) -> tuple[list[np.ndarray[Any, Any]], list[str], list[np.ndarray[Any, Any]], list[str]]:
    encoder = HDCEncoder(dim=_DIM, seed=21)
    rng = np.random.default_rng(99)
    proto_a = encoder.item("class-a")
    proto_b = encoder.item("class-b")
    train = _noisy_copies(rng, proto_a, 8, flip_fraction) + _noisy_copies(
        rng, proto_b, 8, flip_fraction
    )
    train_labels = ["a"] * 8 + ["b"] * 8
    test = _noisy_copies(rng, proto_a, 4, flip_fraction) + _noisy_copies(
        rng, proto_b, 4, flip_fraction
    )
    test_labels = ["a"] * 4 + ["b"] * 4
    return train, train_labels, test, test_labels


def test_fit_and_predict_separate_well_separated_clusters_exactly() -> None:
    train, train_labels, test, test_labels = _two_cluster_problem(0.10)
    clf = CentroidHDClassifier(HDCEncoder(dim=_DIM, seed=21))
    clf.fit(train, train_labels)
    assert clf.classes == ("a", "b")
    predictions = [clf.predict(vector) for vector in test]
    assert predictions == test_labels


def test_retrain_reduces_mistakes_and_never_worsens_training_accuracy() -> None:
    train, train_labels, _, _ = _two_cluster_problem(0.35)
    clf = CentroidHDClassifier(HDCEncoder(dim=_DIM, seed=21))
    clf.fit(train, train_labels)
    mistakes = clf.retrain(train, train_labels, epochs=5)
    assert len(mistakes) == 5
    assert mistakes[-1] <= mistakes[0]
    final = sum(1 for vector, label in zip(train, train_labels) if clf.predict(vector) != label)
    assert final <= mistakes[0]


def test_retrain_applies_the_mistake_update_and_corrects_the_example() -> None:
    clf = CentroidHDClassifier(HDCEncoder(dim=8, seed=1))
    proto_a = np.array([1, 1, 1, 1, 0, 0, 0, 0], dtype=np.uint8)
    proto_b = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.uint8)
    clf.fit([proto_a, proto_b], ["a", "b"])
    hard_example = np.array([0, 0, 1, 1, 1, 1, 1, 1], dtype=np.uint8)
    assert clf.predict(hard_example) == "b"
    mistakes = clf.retrain([hard_example], ["a"], epochs=3)
    assert mistakes[0] == 1
    assert mistakes[-1] == 0
    assert clf.predict(hard_example) == "a"


def test_seeded_pipeline_is_deterministic() -> None:
    train, train_labels, test, _ = _two_cluster_problem(0.20)
    runs = []
    for _ in range(2):
        clf = CentroidHDClassifier(HDCEncoder(dim=_DIM, seed=21, tie_policy="random"))
        clf.fit(train, train_labels)
        clf.retrain(train, train_labels, epochs=2)
        runs.append([clf.predict(vector) for vector in test])
    assert runs[0] == runs[1]


def test_centroid_returns_defensive_copy() -> None:
    train, train_labels, _, _ = _two_cluster_problem(0.10)
    clf = CentroidHDClassifier(HDCEncoder(dim=_DIM, seed=21))
    clf.fit(train, train_labels)
    centroid = clf.centroid("a")
    centroid[0] ^= 1
    assert not np.array_equal(centroid, clf.centroid("a"))


def test_fit_rejects_empty_or_mismatched_inputs() -> None:
    clf = CentroidHDClassifier(HDCEncoder(dim=_DIM, seed=1))
    with pytest.raises(ValueError, match="vectors and labels"):
        clf.fit([], [])
    with pytest.raises(ValueError, match="vectors and labels"):
        clf.fit([np.zeros(_DIM, dtype=np.uint8)], ["a", "b"])


def test_fit_rejects_wrong_shape_and_non_binary_vectors() -> None:
    clf = CentroidHDClassifier(HDCEncoder(dim=_DIM, seed=1))
    with pytest.raises(ValueError, match="shape"):
        clf.fit([np.zeros(_DIM - 1, dtype=np.uint8)], ["a"])
    with pytest.raises(ValueError, match="binary"):
        clf.fit([np.full(_DIM, 2, dtype=np.uint8)], ["a"])


def test_predict_requires_fitted_classes() -> None:
    clf = CentroidHDClassifier(HDCEncoder(dim=_DIM, seed=1))
    with pytest.raises(ValueError, match="no fitted classes"):
        clf.predict(np.zeros(_DIM, dtype=np.uint8))


def test_retrain_rejects_invalid_epochs_inputs_and_unknown_labels() -> None:
    train, train_labels, _, _ = _two_cluster_problem(0.10)
    clf = CentroidHDClassifier(HDCEncoder(dim=_DIM, seed=21))
    clf.fit(train, train_labels)
    with pytest.raises(ValueError, match="epochs"):
        clf.retrain(train, train_labels, epochs=0)
    with pytest.raises(ValueError, match="epochs"):
        clf.retrain(train, train_labels, epochs=cast(int, 2.0))
    with pytest.raises(ValueError, match="vectors and labels"):
        clf.retrain([], [], epochs=1)
    with pytest.raises(ValueError, match="unknown"):
        clf.retrain(train, ["z"] * len(train), epochs=1)


def test_centroid_rejects_unknown_class() -> None:
    clf = CentroidHDClassifier(HDCEncoder(dim=_DIM, seed=1))
    with pytest.raises(ValueError, match="unknown class"):
        clf.centroid("missing")


def test_all_zero_accumulator_tie_resolves_through_encoder_policy() -> None:
    encoder = HDCEncoder(dim=8, seed=3, tie_policy="ones")
    clf = CentroidHDClassifier(encoder)
    vector = np.array([1, 0, 1, 0, 1, 0, 1, 0], dtype=np.uint8)
    inverse = np.bitwise_xor(vector, 1)
    clf.fit([vector, inverse], ["a", "a"])
    assert np.array_equal(clf.centroid("a"), np.ones(8, dtype=np.uint8))

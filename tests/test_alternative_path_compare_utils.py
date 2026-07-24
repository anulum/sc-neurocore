# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (compare_utils) from former test_alternative_path.py

from __future__ import annotations

from tests.alternative_path_support import *  # noqa: F403


def test_compare_numeric_shape_mismatch_is_reported_as_diverged():
    stats = compare_outputs(
        np.array([1.0, 2.0]), np.array([1.0, 2.0, 3.0]), AlternativePathConfig()
    )
    assert not stats.matched
    assert "shape mismatch" in stats.detail


def test_compare_empty_numeric_arrays_match_trivially():
    stats = compare_outputs(np.array([]), np.array([]), AlternativePathConfig())
    assert stats.matched
    assert stats.comparable_leaf_count == 1
    assert stats.max_abs_diff == 0.0
    assert "empty numeric outputs matched" in stats.detail


def test_compare_empty_mappings_have_no_comparable_leaves():
    stats = compare_outputs({}, {}, AlternativePathConfig())
    assert stats.matched
    assert stats.comparable_leaf_count == 0
    assert "no comparable leaves" in stats.detail


def test_compare_mappings_with_different_keys_diverge():
    stats = compare_outputs({"a": 1}, {"b": 1}, AlternativePathConfig())
    assert not stats.matched
    assert "mapping keys differ" in stats.detail


def test_compare_ragged_sequences_of_unequal_length_diverge():
    # A ragged baseline makes ``np.asarray`` raise, so the numeric comparator
    # returns None and the sequence branch handles the length mismatch.
    stats = compare_outputs([1, [2, 3]], [1], AlternativePathConfig())
    assert not stats.matched
    assert "sequence length mismatch" in stats.detail


def test_compare_mixed_sequences_combine_per_element_matches():
    # Mixed numeric/string sequences are non-numeric to ``np.asarray`` (string
    # dtype), so each element is compared recursively and combined.
    stats = compare_outputs([1, "tag"], [1, "tag"], AlternativePathConfig())
    assert stats.matched
    assert stats.comparable_leaf_count == 2

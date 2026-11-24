# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPurePythonFallbacks from former test_variability_edge_cases.py

"""Focused suite: TestPurePythonFallbacks from former test_variability_edge_cases.py."""

from __future__ import annotations

from sc_neurocore.analysis.spike_stats import variability as variability_module
from sc_neurocore.analysis.spike_stats.variability import approximate_entropy
from sc_neurocore.analysis.spike_stats.variability import lempel_ziv_complexity
from sc_neurocore.analysis.spike_stats.variability import permutation_entropy
from sc_neurocore.analysis.spike_stats.variability import sample_entropy
from tests.variability_edge_cases_support import _RUST_AVAILABLE
from tests.variability_edge_cases_support import _bernoulli_train
import numpy as np
import pytest
from tests.variability_edge_cases_support import force_python_fallback as force_python_fallback


class TestPurePythonFallbacks:
    """The pure-Python reference paths of the variability measures.

    Exercise the reference Python implementations that shadow the Rust core,
    and confirm they agree with the Rust path when it is available.
    """

    def test_lempel_ziv_python_branch(self, force_python_fallback: None) -> None:
        """The Python Lempel-Ziv complexity of a Bernoulli train is finite and positive."""
        value = lempel_ziv_complexity(_bernoulli_train(0.3, 256, 7))
        assert np.isfinite(value) and value > 0.0

    def test_approximate_entropy_python_branch(self, force_python_fallback: None) -> None:
        """The Python approximate entropy of a Bernoulli train is finite."""
        value = approximate_entropy(_bernoulli_train(0.3, 200, 11))
        assert np.isfinite(value)

    def test_sample_entropy_python_branch(self, force_python_fallback: None) -> None:
        """The Python sample entropy of a Bernoulli train is computed (the check only excludes infinity)."""
        value = sample_entropy(_bernoulli_train(0.35, 200, 13))
        assert np.isfinite(value) or np.isnan(value)

    def test_sample_entropy_no_matches_returns_nan(self, force_python_fallback: None) -> None:
        """The Python sample entropy without template matches is NaN."""
        # Alternating 0/1 of length 4: the two length-2 templates differ by 1,
        # which exceeds r = 0.2*std, so no template pair matches and b == 0.
        train = np.array([0, 1, 0, 1], dtype=np.int8)
        assert np.isnan(sample_entropy(train, m=2))

    def test_permutation_entropy_python_branch(self, force_python_fallback: None) -> None:
        """The Python permutation entropy of a Bernoulli train lies inside [0, 1]."""
        value = permutation_entropy(_bernoulli_train(0.4, 200, 17), order=3)
        assert 0.0 <= value <= 1.0

    def test_permutation_entropy_order_one_degenerate(self, force_python_fallback: None) -> None:
        """The Python permutation entropy of order one is zero."""
        # order=1 -> a single ordinal pattern -> h_max = log2(1!) = 0 -> 0.0.
        value = permutation_entropy(_bernoulli_train(0.4, 100, 19), order=1)
        assert value == 0.0

    @pytest.mark.skipif(not _RUST_AVAILABLE, reason="Rust spike_stats_core not built")
    def test_lempel_ziv_python_matches_rust(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The Python and Rust Lempel-Ziv complexities agree within 1e-9."""
        train = _bernoulli_train(0.3, 256, 7)
        rust_value = lempel_ziv_complexity(train)
        monkeypatch.setattr(variability_module, "_HAS_RUST", False)
        python_value = lempel_ziv_complexity(train)
        assert np.isclose(rust_value, python_value, rtol=1e-9, atol=0.0)

    @pytest.mark.skipif(not _RUST_AVAILABLE, reason="Rust spike_stats_core not built")
    def test_entropies_python_match_rust(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The Python and Rust approximate, sample and permutation entropies agree within 1e-6."""
        train = _bernoulli_train(0.35, 200, 23)
        rust = (
            approximate_entropy(train),
            sample_entropy(train),
            permutation_entropy(train, order=3),
        )
        monkeypatch.setattr(variability_module, "_HAS_RUST", False)
        python = (
            approximate_entropy(train),
            sample_entropy(train),
            permutation_entropy(train, order=3),
        )
        for rust_value, python_value in zip(rust, python):
            assert np.isclose(rust_value, python_value, rtol=1e-6, atol=1e-9, equal_nan=True)

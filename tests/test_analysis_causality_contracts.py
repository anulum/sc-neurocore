# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Contract tests for spike-train causality analysis

"""Strict contract tests for spike-train causality analysis."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

import sc_neurocore.analysis.spike_stats.causality as causality_module
from sc_neurocore.analysis.spike_stats.causality import (
    conditional_granger_causality,
    directed_transfer_function,
    pairwise_granger_causality,
    spectral_granger_causality,
)


def _binary_train(length: int = 48) -> npt.NDArray[np.float64]:
    """Return a deterministic non-degenerate binary spike train."""
    train = np.zeros(length, dtype=np.float64)
    train[3::7] = 1.0
    train[5::11] = 1.0
    return train


def test_pairwise_granger_rejects_invalid_domains() -> None:
    """Reject invalid binning, model order, and non-finite source values."""
    train = _binary_train()

    with pytest.raises(ValueError, match="bin_size"):
        pairwise_granger_causality(train, train, bin_size=0, order=2)

    with pytest.raises(ValueError, match="order"):
        pairwise_granger_causality(train, train, bin_size=1, order=0)

    invalid = train.copy()
    invalid[8] = np.nan
    with pytest.raises(ValueError, match="finite"):
        pairwise_granger_causality(invalid, train, bin_size=1, order=2)

    nonnumeric = np.array(["not-a-spike"], dtype=object)
    with pytest.raises(ValueError, match="numeric"):
        pairwise_granger_causality(nonnumeric, train, bin_size=1, order=2)

    matrix_train = train.reshape(6, 8)
    with pytest.raises(ValueError, match="one-dimensional"):
        pairwise_granger_causality(matrix_train, train, bin_size=1, order=2)


def test_conditional_granger_rejects_invalid_condition_values() -> None:
    """Reject invalid conditional Granger control traces before regression."""
    train = _binary_train()
    condition = train.copy()
    condition[4] = np.inf

    with pytest.raises(ValueError, match="finite"):
        conditional_granger_causality(train, train, condition, bin_size=1, order=2)


def test_frequency_domain_measures_reject_invalid_domains() -> None:
    """Reject invalid population and frequency-domain contracts."""
    trains = [_binary_train(), _binary_train()]

    with pytest.raises(ValueError, match="n_freqs"):
        spectral_granger_causality(trains, bin_size=1, order=2, n_freqs=0)

    with pytest.raises(ValueError, match="at least one"):
        directed_transfer_function([], bin_size=1, order=2, n_freqs=4)

    with pytest.raises(ValueError, match="same number of bins"):
        spectral_granger_causality(
            [_binary_train(48), _binary_train(40)], bin_size=1, order=2, n_freqs=4
        )


def test_zero_residual_granger_returns_no_signal() -> None:
    """All-zero binned targets produce no directed Granger signal."""
    zero = np.zeros(64, dtype=np.float64)

    assert pairwise_granger_causality(zero, zero, bin_size=1, order=2) == 0.0
    assert conditional_granger_causality(zero, zero, zero, bin_size=1, order=2) == 0.0


def test_short_population_uses_stable_var_fallback() -> None:
    """Very short populations return finite zero-valued frequency tensors."""
    trains = [np.array([0.0, 1.0]), np.array([1.0, 0.0])]

    spectrum = spectral_granger_causality(trains, bin_size=1, order=5, n_freqs=3)

    assert spectrum.shape == (2, 2, 3)
    assert np.all(np.isfinite(spectrum))


def test_singular_frequency_transfer_is_skipped(monkeypatch: pytest.MonkeyPatch) -> None:
    """Singular VAR transfer matrices leave spectral GC and DTF at zero."""

    def singular_var(
        trains_binned: npt.NDArray[np.float64], order: int
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        dimension = trains_binned.shape[0]
        if order != 1:
            raise AssertionError("test fixture expects VAR(1)")
        return np.eye(dimension, dtype=np.float64), np.eye(dimension, dtype=np.float64)

    monkeypatch.setattr(causality_module, "_var_coefficients", singular_var)
    trains = [_binary_train(), _binary_train()]

    spectrum = spectral_granger_causality(trains, bin_size=1, order=1, n_freqs=1)
    dtf = directed_transfer_function(trains, bin_size=1, order=1, n_freqs=1)

    assert spectrum.shape == (2, 2, 1)
    assert dtf.shape == (2, 2, 1)
    assert np.array_equal(spectrum, np.zeros_like(spectrum))
    assert np.array_equal(dtf, np.zeros_like(dtf))

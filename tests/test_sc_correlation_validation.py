# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Stochastic-correlation validation tests

"""Fail-closed shape, binary-domain, length, and threshold contracts."""

from __future__ import annotations

import pytest

from sc_neurocore.core import correlation_diagnostic, estimate_scc, observed_and_bias


def test_length_mismatch_rejected() -> None:
    with pytest.raises(ValueError):
        estimate_scc([1, 0, 1], [1, 0])


def test_empty_rejected() -> None:
    with pytest.raises(ValueError):
        estimate_scc([], [])


def test_non_binary_rejected() -> None:
    with pytest.raises(ValueError):
        estimate_scc([0, 2, 1], [0, 1, 1])


def test_non_1d_rejected() -> None:
    with pytest.raises(ValueError):
        estimate_scc([[1, 0], [0, 1]], [[1, 0], [0, 1]])


def test_observed_bias_length_mismatch_rejected() -> None:
    with pytest.raises(ValueError):
        observed_and_bias([1, 0, 1], [1, 0])


def test_diagnostic_length_mismatch_rejected() -> None:
    with pytest.raises(ValueError):
        correlation_diagnostic([1, 0, 1], [1, 0])


def test_diagnostic_negative_threshold_rejected() -> None:
    with pytest.raises(ValueError):
        correlation_diagnostic([1, 0, 1], [0, 1, 1], bias_threshold=-0.1)

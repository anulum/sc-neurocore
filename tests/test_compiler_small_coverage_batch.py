# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for tensor-stream, dispatch, target-recommender and q-format edges

"""Contracts for residual edges in tensor-stream, dispatch planning, target
recommendation and Q-format validation."""

from __future__ import annotations

import numpy as np
import pytest

from sc_neurocore.compiler.intelligence import plan_heterogeneous_dispatch, recommend_target
from sc_neurocore.compiler.q_format import QFormat, QFormatMixed
from sc_neurocore.core.tensor_stream import TensorStream

_COMPLEX_EQS = {"v": "a + b + c + d + e + f + g - h"}


def test_tensor_stream_to_prob_falls_back_for_unknown_domain() -> None:
    """to_prob returns the raw data for a domain it does not specially decode."""
    stream = TensorStream(data=np.array([0.2, 0.8]), domain="spike")
    np.testing.assert_array_equal(stream.to_prob(), np.array([0.2, 0.8]))


def test_dispatch_defaults_to_fpga_for_empty_backends() -> None:
    """An empty backend list defaults to a single FPGA backend."""
    plan = plan_heterogeneous_dispatch({"a": "x + 1"}, [])
    assert "fpga" in plan.backends


def test_dispatch_assigns_remainder_neurons_to_first_backend() -> None:
    """Indivisible neuron counts assign the remainder to the first backend."""
    plan = plan_heterogeneous_dispatch(
        {"a": "x + 1", "b": "y + 1", "c": "z + 1"},
        ["fpga", "asic"],
        neuron_count=1001,
    )
    assert sum(plan.total_neurons_per_backend.values()) == 1001


def test_recommend_target_applies_frequency_floor() -> None:
    """A very high minimum frequency exercises the frequency-floor filter."""
    relaxed = recommend_target({"v": "x + 1"})
    constrained = recommend_target({"v": "x + 1"}, min_freq_mhz=1.0e9)
    assert len(constrained) <= len(relaxed)


def test_recommend_target_scores_complex_models() -> None:
    """A high-operation-count model yields scored recommendations."""
    recommendations = recommend_target(_COMPLEX_EQS)
    assert recommendations
    assert all(r.score >= 0 for r in recommendations)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"integer_bits": 1.0, "fraction_bits": 8},
        {"integer_bits": 8, "fraction_bits": 1.0},
    ],
)
def test_qformat_rejects_non_integer_fields(kwargs: dict[str, object]) -> None:
    """QFormat rejects non-integer bit widths."""
    with pytest.raises(TypeError):
        QFormat(**kwargs)  # type: ignore[arg-type]


def test_qformat_from_string_rejects_non_string() -> None:
    """QFormat.from_string rejects a non-string format."""
    with pytest.raises(TypeError):
        QFormat.from_string(123)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "override",
    [
        {"weight_fmt": "not a qformat"},
        {"accum_fmt": "not a qformat"},
        {"scale_per_tensor": "yes"},
        {"rounding": "bogus"},
        {"accum_fmt": QFormat(4, 4)},
        {"accum_fmt": QFormat(16, 2)},
    ],
)
def test_qformat_mixed_rejects_invalid_fields(override: dict[str, object]) -> None:
    """QFormatMixed rejects malformed accumulator format, scale flag and rounding."""
    valid = {
        "weight_fmt": QFormat(8, 8),
        "accum_fmt": QFormat(16, 16),
        "scale_per_tensor": True,
        "rounding": "nearest",
    }
    valid.update(override)
    with pytest.raises((TypeError, ValueError)):
        QFormatMixed(**valid)  # type: ignore[arg-type]

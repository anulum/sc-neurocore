# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

from __future__ import annotations

import math

import pytest

from sc_neurocore.security.side_channel_metrics import (
    SideChannelMetricError,
    compute_class_activity_proxy,
    compute_switching_activity,
)


def test_switching_activity_reports_per_stream_rates_and_histogram() -> None:
    summary = compute_switching_activity(
        (
            (0, 0, 1, 1),
            (1, 0, 0, 1),
            (1, 1, 1, 1),
        )
    )

    assert summary.stream_count == 3
    assert summary.cycles == 4
    assert summary.per_stream_transition_counts == (1, 2, 0)
    assert summary.per_stream_transition_rates == (
        pytest.approx(1.0 / 3.0),
        pytest.approx(2.0 / 3.0),
        0.0,
    )
    assert summary.total_transitions == 3
    assert summary.mean_transition_rate == pytest.approx(1.0 / 3.0)
    assert summary.max_transition_rate == pytest.approx(2.0 / 3.0)
    assert summary.activity_histogram == {0: 1, 1: 1, 2: 1}


@pytest.mark.parametrize(
    "bitstreams",
    [
        (),
        b"\x00\x01",
        ((0, 1), (1,)),
        ((0, 1, 2),),
        ((0, 1.0),),
        ((0, True),),
        ((0,),),
    ],
)
def test_switching_activity_rejects_invalid_stream_matrices(bitstreams: object) -> None:
    with pytest.raises(SideChannelMetricError):
        compute_switching_activity(bitstreams)  # type: ignore[arg-type]


def test_class_activity_proxy_quantifies_label_activity_separation() -> None:
    proxy = compute_class_activity_proxy(
        (
            ((0, 0, 0, 0), (1, 1, 1, 1)),
            ((0, 1, 0, 1), (1, 0, 1, 0)),
            ((1, 1, 1, 1), (0, 0, 0, 0)),
            ((1, 0, 1, 0), (0, 1, 0, 1)),
        ),
        (0, 1, 0, 1),
    )

    assert proxy.sample_count == 4
    assert proxy.class_means == {0: 0.0, 1: 1.0}
    assert proxy.max_class_mean_gap == 1.0
    assert proxy.label_activity_correlation == pytest.approx(1.0)


def test_class_activity_proxy_handles_zero_variance_without_fabricated_correlation() -> None:
    proxy = compute_class_activity_proxy(
        (
            ((0, 1, 0, 1),),
            ((1, 0, 1, 0),),
        ),
        (0, 1),
    )

    assert proxy.class_means == {0: 1.0, 1: 1.0}
    assert proxy.max_class_mean_gap == 0.0
    assert proxy.label_activity_correlation is None


def test_class_activity_proxy_handles_constant_labels_without_fabricated_correlation() -> None:
    proxy = compute_class_activity_proxy(
        (
            ((0, 0, 0, 1),),
            ((0, 1, 0, 1),),
            ((1, 1, 0, 1),),
        ),
        (7, 7, 7),
    )

    assert proxy.class_means == {7: pytest.approx(2.0 / 3.0)}
    assert proxy.max_class_mean_gap == 0.0
    assert proxy.label_activity_correlation is None


def test_class_activity_proxy_rejects_mismatched_or_nonfinite_inputs() -> None:
    with pytest.raises(SideChannelMetricError):
        compute_class_activity_proxy((((0, 1),),), (0, 1))

    with pytest.raises(SideChannelMetricError):
        compute_class_activity_proxy((((0, 1),),), (math.nan,))

    with pytest.raises(SideChannelMetricError):
        compute_class_activity_proxy((((0, 1),),), (False,))


@pytest.mark.parametrize(
    ("samples", "labels"),
    [
        ("invalid", (0,)),
        (b"invalid", (0,)),
        ((((0, 1),),), "invalid"),
        (((("not-a-row",),),), (0,)),
        (((b"\x00\x01",),), (0,)),
    ],
)
def test_class_activity_proxy_rejects_string_like_contract_inputs(samples, labels) -> None:
    with pytest.raises(SideChannelMetricError):
        compute_class_activity_proxy(samples, labels)  # type: ignore[arg-type]

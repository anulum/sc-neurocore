# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore

from __future__ import annotations

import math

import pytest

from tests.performance_guard import (
    STRICT_THROUGHPUT_ENV,
    assert_load_tolerant_throughput,
)


def test_load_tolerant_guard_uses_capped_one_percent_smoke_floor(monkeypatch) -> None:
    monkeypatch.delenv(STRICT_THROUGHPUT_ENV, raising=False)

    assert_load_tolerant_throughput(
        label="shared host",
        observed_per_second=101.0,
        strict_minimum_per_second=50_000.0,
    )

    with pytest.raises(AssertionError, match="smoke guard failed"):
        assert_load_tolerant_throughput(
            label="shared host",
            observed_per_second=100.0,
            strict_minimum_per_second=50_000.0,
        )


def test_load_tolerant_guard_preserves_strict_threshold(monkeypatch) -> None:
    monkeypatch.setenv(STRICT_THROUGHPUT_ENV, "true")

    with pytest.raises(AssertionError, match="throughput regressed"):
        assert_load_tolerant_throughput(
            label="isolated core",
            observed_per_second=49_999.0,
            strict_minimum_per_second=50_000.0,
        )

    assert_load_tolerant_throughput(
        label="isolated core",
        observed_per_second=50_001.0,
        strict_minimum_per_second=50_000.0,
    )


@pytest.mark.parametrize("minimum", [0.0, -1.0, math.inf, math.nan])
def test_load_tolerant_guard_rejects_invalid_strict_minimum(minimum: float) -> None:
    with pytest.raises(AssertionError, match="strict throughput minimum"):
        assert_load_tolerant_throughput(
            label="invalid threshold",
            observed_per_second=1.0,
            strict_minimum_per_second=minimum,
        )

# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore

"""Throughput assertions for load-tolerant functional tests."""

from __future__ import annotations

import math
import os


STRICT_THROUGHPUT_ENV = "SC_NEUROCORE_STRICT_THROUGHPUT"
DEFAULT_SMOKE_FRACTION = 0.01
DEFAULT_SMOKE_CEILING_PER_SECOND = 100.0


def strict_throughput_enabled() -> bool:
    return os.environ.get(STRICT_THROUGHPUT_ENV, "").lower() in {"1", "true", "yes", "on"}


def assert_throughput_guard(
    *,
    label: str,
    observed_per_second: float,
    strict_minimum_per_second: float,
    smoke_minimum_per_second: float,
) -> None:
    """Assert throughput without making non-isolated hosts fail strict gates."""

    assert math.isfinite(observed_per_second) and observed_per_second > 0.0, (
        f"{label} throughput is not finite positive: {observed_per_second!r}"
    )
    if strict_throughput_enabled():
        assert observed_per_second > strict_minimum_per_second, (
            f"{label} throughput regressed: "
            f"{observed_per_second:.0f}/s <= {strict_minimum_per_second:.0f}/s"
        )
        return

    assert observed_per_second > smoke_minimum_per_second, (
        f"{label} throughput smoke guard failed under non-strict local mode: "
        f"{observed_per_second:.0f}/s <= {smoke_minimum_per_second:.0f}/s. "
        f"Set {STRICT_THROUGHPUT_ENV}=1 only on isolated benchmark cores."
    )


def assert_load_tolerant_throughput(
    *,
    label: str,
    observed_per_second: float,
    strict_minimum_per_second: float,
) -> None:
    """Preserve an isolated-core threshold while tolerating shared-host load."""

    assert math.isfinite(strict_minimum_per_second) and strict_minimum_per_second > 0.0, (
        f"{label} strict throughput minimum is not finite positive: {strict_minimum_per_second!r}"
    )
    smoke_minimum = min(
        DEFAULT_SMOKE_CEILING_PER_SECOND,
        strict_minimum_per_second * DEFAULT_SMOKE_FRACTION,
    )
    assert_throughput_guard(
        label=label,
        observed_per_second=observed_per_second,
        strict_minimum_per_second=strict_minimum_per_second,
        smoke_minimum_per_second=smoke_minimum,
    )


def assert_speedup_guard(
    *,
    label: str,
    baseline_seconds: float,
    candidate_seconds: float,
    strict_minimum_speedup: float,
    smoke_minimum_speedup: float,
) -> None:
    """Assert a relative speedup without claiming shared-host benchmark fidelity."""

    for timing_label, timing in (
        ("baseline", baseline_seconds),
        ("candidate", candidate_seconds),
    ):
        assert math.isfinite(timing) and timing > 0.0, (
            f"{label} {timing_label} timing is not finite positive: {timing!r}"
        )
    assert math.isfinite(strict_minimum_speedup) and strict_minimum_speedup > 0.0
    assert math.isfinite(smoke_minimum_speedup) and smoke_minimum_speedup > 0.0
    assert smoke_minimum_speedup <= strict_minimum_speedup

    observed_speedup = baseline_seconds / candidate_seconds
    if strict_throughput_enabled():
        assert observed_speedup > strict_minimum_speedup, (
            f"{label} speedup regressed: {observed_speedup:.2f}x <= {strict_minimum_speedup:.2f}x"
        )
        return

    assert observed_speedup > smoke_minimum_speedup, (
        f"{label} speedup smoke guard failed under non-strict local mode: "
        f"{observed_speedup:.2f}x <= {smoke_minimum_speedup:.2f}x. "
        f"Set {STRICT_THROUGHPUT_ENV}=1 only on isolated benchmark cores."
    )

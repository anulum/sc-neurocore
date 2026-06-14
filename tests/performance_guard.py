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

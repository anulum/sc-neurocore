# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for benchmark measurement context

from __future__ import annotations

from benchmarks._benchmark_context import measurement_context


def test_measurement_context_records_reproducibility_contract() -> None:
    """Benchmark artefacts must preserve isolation and host-load evidence."""

    load_before = [1.0, 2.0, 3.0]
    context = measurement_context(load_before)

    assert context["load_average_before"] == load_before
    assert "taskset affinity only" in str(context["cpu_isolation"])
    assert "non-exclusive workstation run" in str(context["concurrent_load_status"])
    assert "local regression context" in str(context["timing_interpretation"])
    assert "reserved isolated cores" in str(context["production_rerun_requirement"])
    assert "cpu_affinity" in context
    assert "load_average_after" in context
    assert "cpu_governor" in context
    assert "cpu_frequency_mhz" in context

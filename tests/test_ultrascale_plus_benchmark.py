# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Actual UltraScale+ benchmark capture

"""Exercise the target manifest benchmark and its measured physical host context."""

from __future__ import annotations

import os

from benchmarks.bench_ultrascale_plus_target import run


def test_target_manifest_benchmark_reports_hardware_and_shared_host_measurement() -> None:
    """Generate both target manifests without inferring a reservation from affinity."""
    result = run(2, 2)
    assert result["benchmark_isolation_mode"] == "shared-host"
    assert result["host_context"]["runtime_cpuset_shield_claimed"] is False
    assert result["hardware_measurement_claimed"] is False
    assert result["manifest_count"] == 2
    assert result["median_ns_per_manifest"] > 0
    assert len(result["results_ns_per_manifest"]) == 2
    assert all(value > 0 for value in result["results_ns_per_manifest"])
    context = result["measurement_context"]
    assert context["cpu_model"]
    if hasattr(os, "sysconf"):
        assert context["ram_bytes"] == os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    else:
        assert context["ram_bytes"] is None
    assert context["operating_system"]
    assert context["python_version"]

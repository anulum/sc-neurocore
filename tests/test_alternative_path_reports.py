# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (reports) from former test_alternative_path.py

"""Tests of the alternative-path report files."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sc_neurocore.experimental import (
    AlternativePathCase,
    AlternativePathConfig,
    AlternativePathMode,
    AlternativePathRoute,
    build_demo_registry,
    default_report_path,
    write_batch_report,
)


def test_write_batch_report_writes_json_file(tmp_path: Path) -> None:
    """A batch report is written as JSON."""
    registry = build_demo_registry()
    summary = registry.evaluate(
        "demo.affine-sigmoid",
        [AlternativePathCase("small", args=([0.0, 1.0, -1.0],))],
        AlternativePathConfig(enabled=True, mode=AlternativePathMode.SHADOW),
    )
    out = tmp_path / "report.json"

    written = write_batch_report(summary, out)

    assert written == out
    text = out.read_text()
    assert '"route_name": "demo.affine-sigmoid"' in text


def test_default_report_path_is_under_benchmarks_results() -> None:
    """The default report path lies under benchmarks/results."""
    path = default_report_path("physics.heat.cosine-mode")

    assert str(path) == "benchmarks/results/experimental_physics_heat_cosine_mode.json"


def test_default_report_path_handles_second_physics_route() -> None:
    """A second physics route gets its own default path."""
    path = default_report_path("physics.oscillator.harmonic-symplectic")

    assert (
        str(path) == "benchmarks/results/experimental_physics_oscillator_harmonic_symplectic.json"
    )


def test_default_report_path_handles_solver_route() -> None:
    """A solver route gets its own default path."""
    path = default_report_path("solver.lif.subthreshold-exact")

    assert str(path) == "benchmarks/results/experimental_solver_lif_subthreshold_exact.json"


def test_result_report_is_json_friendly() -> None:
    """A result report holds only JSON values."""
    route = AlternativePathRoute(
        name="safe.report",
        baseline=lambda: {"x": np.array([1.0, 2.0])},
        candidate=lambda: {"x": np.array([1.0, 2.0])},
        summary="Report route",
        expected_behavior="Serializable report should not leak raw arrays",
    )

    result = route.run(
        AlternativePathConfig(enabled=True, mode=AlternativePathMode.SHADOW),
    )
    report = result.to_report()

    assert report["route_name"] == "safe.report"
    assert report["returned_path"] == "shadow-baseline"
    assert report["comparison"]["matched"] is True

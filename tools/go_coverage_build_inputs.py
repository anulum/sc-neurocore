# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Validation of actual native build and test observations

"""Validate real compiler observations separately from launching the pinned native tools."""

from __future__ import annotations

from typing import cast

from tools.go_coverage_profiles import MODULE


def validate_gofmt_output(output: str) -> None:
    """Refuse every actual formatter listing before accepting the native build inputs."""
    if output.strip():
        raise RuntimeError("A new Go coverage source is not formatted")


def validate_native_exports(model: str, variant: str, symbols: str) -> None:
    """Conserve the real simulation export and the existing diagnostic build-tag boundary."""
    names = {line.split()[-1] for line in symbols.splitlines() if line.split()}
    if model + "_simulate_c" not in names:
        raise RuntimeError("An actual MAT simulation export is missing")
    diagnostic = any("coverage_snapshot" in name for name in names)
    if diagnostic != (variant != "normal"):
        raise RuntimeError("Diagnostic export does not respect the build tag")


def validate_native_dependencies(tagged: bool, dependencies: list[str]) -> None:
    """Require both real coverage packages exactly when the producer enables its build tag."""
    for package in (MODULE + "internal/coverageprofile", "runtime/coverage"):
        if (package in dependencies) != tagged:
            raise RuntimeError("Native coverage package does not respect the build tag")


def validate_native_test_names(expected: list[str]) -> None:
    """Keep the complete original nine and added four source-declared native test cohort."""
    if len(expected) != 13 or len(set(expected)) != 13:
        raise RuntimeError("The complete original9 and added4 native test cohort changed")


def validate_native_test_events(expected: list[str], events: list[dict[str, object]]) -> list[str]:
    """Accept only the complete genuine Go JSON pass cohort without failed or skipped events."""
    if any(event.get("Action") in {"fail", "skip"} for event in events):
        raise RuntimeError("A maintained native family test failed or skipped")
    passed = [
        event["Test"] for event in events if event.get("Action") == "pass" and "Test" in event
    ]
    if (
        len(passed) != len(expected)
        or any(not isinstance(name, str) for name in passed)
        or set(passed) != set(expected)
    ):
        raise RuntimeError("Native compilation did not execute the full family cohort")
    return cast(list[str], passed)

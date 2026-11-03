# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Complete native producer input and final custody validation

"""Validate actual native producer observations without replacing compiler or model execution."""

from __future__ import annotations

from pathlib import Path
import re
from typing import TypedDict

from tools.go_coverage_profiles import TARGET_STATEMENTS
from tools.go_mat_coverage_build import digest


class BaselineSource(TypedDict):
    """Describe the genuine public profile CLI's per-source before-work observation."""

    path: str
    covered_statements: int
    total_statements: int
    present: bool


def validate_image_transfer(original: Path, copied: Path, before: str) -> None:
    """Read both real files again and preserve the complete producer digest across transfer."""
    if digest(original) != before or digest(copied) != before:
        raise RuntimeError("Same-job Mojo image changed during input transfer")


def validate_compiler_version(module: str, observed: str) -> None:
    """Require the actual Go version stdout to match the maintained exact module declaration."""
    version = re.search(r"^go (\d+\.\d+\.\d+)$", module, re.MULTILINE)
    if version is None or not observed.startswith("go version go" + version[1] + " "):
        raise RuntimeError("Actual Go compiler differs from the maintained module pin")


def validate_before_work_sources(rows: list[BaselineSource]) -> None:
    """Keep every original native subject present, unshrunk and genuinely zero before work."""
    if len(rows) != len(TARGET_STATEMENTS) or any(
        row["covered_statements"] != 0
        or not row["present"]
        or row["total_statements"] < TARGET_STATEMENTS[row["path"]]
        for row in rows
    ):
        raise RuntimeError(
            "Actual before-work native subjects are missing, shrunk or already covered"
        )


def validate_final_images(images: list[dict[str, str]]) -> None:
    """Read the complete declared backing files again before claiming unchanged final custody."""
    if any(digest(Path(row["path"])) != row["sha256"] for row in images):
        raise RuntimeError("A declared actual native image changed during qualification")


def validate_bootstrap_custody(before: bytes, after: bytes) -> None:
    """Require the original exact serialized bootstrap declarations to remain byte-identical."""
    if before != after:
        raise RuntimeError("An actual gate-owned child bootstrap changed during qualification")


def validate_source_custody(before: dict[str, str], after: dict[str, str]) -> None:
    """Conserve all first-party source and producer executable pins without omitting entries."""
    if before != after:
        raise RuntimeError(
            "A first-party source or producer executable changed during qualification"
        )

# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Schema validator contract test support

"""Shared schema builders and error selectors for validator contract tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sc_neurocore.neurons.schema_validator import SchemaError


def _valid_schema() -> dict[str, Any]:
    """Return a minimal schema dictionary that passes structural validation."""
    return {
        "metadata": {"schema_version": 1, "name": "demo"},
        "state": {"v": 0.0},
        "dynamics": {"v": "v + I"},
        "integration": {"dt": 0.1, "method": "euler"},
    }


def _levels(errors: list[SchemaError], message_fragment: str) -> list[SchemaError]:
    """Return errors whose message contains the given fragment."""
    return [error for error in errors if message_fragment in error.message]


def _write_schema_files(
    directory: Path,
    name: str,
    *,
    toml: str | None,
    data: dict[str, Any] | None,
) -> None:
    """Write optional TOML and JSON variants into a schema directory."""
    if toml is not None:
        (directory / f"{name}.toml").write_text(toml, encoding="utf-8")
    if data is not None:
        (directory / f"{name}.json").write_text(json.dumps(data), encoding="utf-8")

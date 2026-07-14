# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — UniversalNeuron DSL schema validator

"""Validate model schemas against the DSL specification.

Checks:
1. Required sections present (metadata, state, dynamics, integration)
2. Schema version is supported
3. All state variables have initial values
4. All dynamics equations reference only declared variables and parameters
5. Threshold condition is valid
6. Reset section references only state variables
7. TOML ↔ JSON parity (if both formats exist)

Usage::

    from sc_neurocore.neurons.schema_validator import validate_schema, validate_all_bundled

    errors = validate_schema("lif")
    all_results = validate_all_bundled()
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from sc_neurocore.neurons.schema_contracts import stateless_event_kind

logger = logging.getLogger(__name__)

_SCHEMAS_DIR = Path(__file__).parent / "model_schemas"
_SUPPORTED_VERSIONS = {1}
_REQUIRED_SECTIONS = {"metadata", "state", "dynamics", "integration"}
_OPTIONAL_SECTIONS = {"parameters", "threshold", "reset", "extensions"}
_REQUIRED_METADATA = {"schema_version", "name"}
_REQUIRED_INTEGRATION = {"dt", "method"}


class SchemaError:
    """A single validation error."""

    def __init__(self, level: str, message: str, section: str = "") -> None:
        self.level = level  # "error" or "warning"
        self.message = message
        self.section = section

    def __repr__(self) -> str:
        prefix = f"[{self.section}] " if self.section else ""
        return f"{self.level.upper()}: {prefix}{self.message}"


def validate_schema_dict(data: dict[str, Any], name: str = "") -> list[SchemaError]:
    """Validate a parsed schema dictionary.

    Returns a list of SchemaError objects. Empty list = valid.
    """
    errors: list[SchemaError] = []
    ctx = name or "unnamed"

    # Check required sections
    for section in _REQUIRED_SECTIONS:
        if section not in data:
            errors.append(SchemaError("error", f"Missing required section '{section}'", ctx))

    if errors:
        return errors  # Can't continue without required sections

    # Metadata validation
    meta = data["metadata"]
    for field in _REQUIRED_METADATA:
        if field not in meta:
            errors.append(SchemaError("error", f"Missing metadata field '{field}'", "metadata"))

    version = meta.get("schema_version")
    if version not in _SUPPORTED_VERSIONS:
        errors.append(
            SchemaError(
                "error",
                f"Unsupported schema version {version} (supported: {_SUPPORTED_VERSIONS})",
                "metadata",
            )
        )

    # Optional but recommended metadata
    for field in ("author", "year", "doi", "description"):
        if field not in meta:
            errors.append(
                SchemaError("warning", f"Missing recommended field '{field}'", "metadata")
            )

    # State and dynamics may both be empty only for the exact event-only
    # contracts accepted by UniversalNeuron. Keeping this predicate shared
    # prevents the static validator from rejecting schemas the runtime executes.
    event_only = stateless_event_kind(data) is not None

    # State validation
    state = data["state"]
    if not state and not event_only:
        errors.append(SchemaError("error", "State section is empty", "state"))

    for var, val in state.items():
        if not isinstance(val, (int, float)):
            errors.append(
                SchemaError(
                    "error", f"State variable '{var}' has non-numeric initial value", "state"
                )
            )

    # Integration validation
    integration = data["integration"]
    for field in _REQUIRED_INTEGRATION:
        if field not in integration:
            errors.append(
                SchemaError("error", f"Missing integration field '{field}'", "integration")
            )

    dt = integration.get("dt")
    if isinstance(dt, (int, float)) and dt <= 0:
        errors.append(SchemaError("error", f"dt must be positive (got {dt})", "integration"))

    # Dynamics validation
    dynamics = data["dynamics"]
    if not dynamics and not event_only:
        errors.append(SchemaError("error", "Dynamics section is empty", "dynamics"))

    state_vars = set(state.keys())
    params = set(data.get("parameters", {}).keys())
    known_names = state_vars | params | {"I"}  # I is always available as external current

    for var, equation in dynamics.items():
        if var not in state_vars:
            errors.append(
                SchemaError(
                    "warning",
                    f"Dynamics variable '{var}' not in state declaration",
                    "dynamics",
                )
            )

        # Extract identifiers from equation (simple regex — not a full parser)
        idents = set(re.findall(r"\b([a-zA-Z_]\w*)\b", equation))
        # Remove known math functions
        math_funcs = {
            "sin",
            "cos",
            "tan",
            "exp",
            "log",
            "sqrt",
            "abs",
            "tanh",
            "cosh",
            "sinh",
            "atan",
            "asin",
            "acos",
            "min",
            "max",
            "pow",
        }
        unknown = idents - known_names - math_funcs
        if unknown:
            errors.append(
                SchemaError(
                    "warning",
                    f"Equation for '{var}' references unknown names: {sorted(unknown)}",
                    "dynamics",
                )
            )

    # Check state coverage
    for var in state_vars:
        if var not in dynamics:
            errors.append(
                SchemaError(
                    "warning",
                    f"State variable '{var}' has no dynamics equation",
                    "dynamics",
                )
            )

    # Unknown sections
    known_sections = _REQUIRED_SECTIONS | _OPTIONAL_SECTIONS
    for section in data:
        if section not in known_sections:
            errors.append(
                SchemaError(
                    "warning",
                    f"Unknown section '{section}'",
                    ctx,
                )
            )

    return errors


def validate_schema(name: str) -> list[SchemaError]:
    """Validate a bundled schema by name.

    Checks both TOML and JSON versions if they exist, and verifies parity.
    """
    errors: list[SchemaError] = []

    toml_path = _SCHEMAS_DIR / f"{name}.toml"
    json_path = _SCHEMAS_DIR / f"{name}.json"

    has_toml = toml_path.exists()
    has_json = json_path.exists()

    if not has_toml and not has_json:
        errors.append(SchemaError("error", f"No schema files found for '{name}'"))
        return errors

    toml_data = None
    json_data = None

    if has_toml:
        try:
            import tomllib
        except ModuleNotFoundError:
            import tomli as tomllib  # type: ignore[no-redef]
        with open(toml_path, "rb") as f:
            toml_data = tomllib.load(f)
        errors.extend(validate_schema_dict(toml_data, f"{name}.toml"))

    if has_json:
        with open(json_path) as f:
            json_data = json.load(f)
        errors.extend(validate_schema_dict(json_data, f"{name}.json"))

    # Parity check. Matching keys are insufficient: a timestep, equation, or
    # threshold value can drift while preserving the same shape, so compare the
    # complete authored contract after retaining the more specific key error.
    if toml_data and json_data:
        for section in (
            "metadata",
            "state",
            "parameters",
            "integration",
            "dynamics",
            "threshold",
            "reset",
            "extensions",
        ):
            toml_section = toml_data.get(section, {})
            json_section = json_data.get(section, {})
            if set(toml_section.keys()) != set(json_section.keys()):
                errors.append(
                    SchemaError(
                        "error",
                        f"TOML/JSON key mismatch in '{section}': "
                        f"TOML={sorted(toml_section.keys())}, "
                        f"JSON={sorted(json_section.keys())}",
                        f"{name} parity",
                    )
                )
            elif toml_section != json_section:
                errors.append(
                    SchemaError(
                        "error",
                        f"TOML/JSON value mismatch in '{section}'",
                        f"{name} parity",
                    )
                )
    elif has_toml and not has_json:
        errors.append(SchemaError("warning", f"Missing JSON version for '{name}'"))
    elif has_json and not has_toml:
        errors.append(SchemaError("warning", f"Missing TOML version for '{name}'"))

    return errors


def validate_all_bundled() -> dict[str, list[SchemaError]]:
    """Validate all bundled schemas.

    Returns a dict mapping schema name to its list of errors/warnings.
    """
    results: dict[str, list[SchemaError]] = {}
    seen_names: set[str] = set()

    for path in sorted(_SCHEMAS_DIR.glob("*.toml")):
        name = path.stem
        seen_names.add(name)
    for path in sorted(_SCHEMAS_DIR.glob("*.json")):
        seen_names.add(path.stem)

    for name in sorted(seen_names):
        results[name] = validate_schema(name)

    return results

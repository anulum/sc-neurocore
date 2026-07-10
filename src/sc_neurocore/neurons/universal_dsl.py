# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Universal Neuron DSL: schema-driven model instantiation

"""Universal Neuron DSL — a parallel meta-layer over existing neuron models.

This module does NOT replace the 149+ hand-crafted neuron model files in
``neurons/models/``.  Each of those files is individually tested, documented,
and traced to its publication.  The Universal DSL provides a schema-driven
alternative path for:

1. Rapid prototyping of new models without writing Python classes
2. Programmatic model generation from Studio / JSON APIs
3. Automated Verilog and Rust code generation from a single source
4. Database-scale model management as the collection grows

Schemas can be written in TOML (human-editable, version-controlled) or JSON
(programmatic, Studio interchange).  The loader accepts both formats.

Schema format (v1):

.. code-block:: toml

    [metadata]
    schema_version = 1
    name = "FitzHugh-Nagumo"
    author = "FitzHugh, R."
    year = 1961
    doi = "10.1016/S0006-3495(61)86902-6"
    description = "2D qualitative spike model"

    [state]
    v = -1.0
    w = -0.5

    [parameters]
    a = 0.7
    b = 0.8
    epsilon = 0.08

    [integration]
    dt = 0.1
    method = "euler"   # euler | map | rk4 | exp_euler
    substeps = 1       # inner integration sub-steps per macro step (default 1). >1 advances
                       # `substeps` sub-steps of `dt` before a single macro-boundary spike
                       # decision, matching a sub-stepping hand model (e.g. 100 dt sub-steps
                       # per 1 ms macro step); the crossing is taken once per macro step.

    [dynamics]
    # ODE right-hand sides: dvar/dt = expression
    v = "v - v**3 / 3 - w + I"
    w = "epsilon * (v + a - b * w)"

    [threshold]
    condition = "v > 1.0"
    detection = "crossing"   # "level" fires on every step the condition holds;
                             # "crossing" fires once on the rising (inactive->active)
                             # edge, for a non-resetting oscillator. A reset that clears
                             # the condition makes the two identical, so reset-based
                             # integrate-and-fire models use "level" semantics either way.

    [reset]
    v = "-1.0"
    w = "w"     # no reset for w (identity)

    [extensions]
    # Forward-compatible: arbitrary key-value pairs for future features
    # e.g. plasticity_rule = "stdp", hardware_target = "ice40"
"""

from __future__ import annotations

import json
import logging
from copy import deepcopy
from pathlib import Path
from typing import Any

from sc_neurocore.neurons.equation_builder import EquationNeuron

logger = logging.getLogger(__name__)

# Schema versions this module understands.
# v1: core sections (metadata/state/parameters/integration/dynamics/threshold/reset/extensions).
# v2: adds the OPTIONAL layered science-knowledge sections (science/validation/provenance/hints)
#     that make the catalogue an auditable scientific record — see the catalogue-to-silicon
#     master plan §4A / invariant I7. Every section is optional, so every v1 schema is already a
#     valid v2 schema and keeps loading unchanged (backward-compatible, no migration).
_SUPPORTED_SCHEMA_VERSIONS = {1, 2}

# Directory containing bundled TOML/JSON schemas
_SCHEMA_DIR = Path(__file__).parent / "model_schemas"


def _load_toml(path: Path) -> dict[str, Any]:
    """Load a TOML file, using tomllib (3.11+) or tomli fallback."""
    import sys

    if sys.version_info >= (3, 11):
        import tomllib
    else:
        try:
            import tomli as tomllib  # type: ignore[no-redef]
        except ImportError as exc:
            raise ImportError(
                "Python <3.11 requires the 'tomli' package for TOML support. "
                "Install with: pip install tomli"
            ) from exc
    with open(path, "rb") as f:
        return tomllib.load(f)


def _load_json(path: Path) -> dict[str, Any]:
    """Load a JSON schema file."""
    with open(path) as f:
        data: dict[str, Any] = json.load(f)
        return data


def load_schema(source: str | Path) -> dict[str, Any]:
    """Load a neuron model schema from a TOML or JSON file.

    Parameters
    ----------
    source : str or Path
        Path to a ``.toml`` or ``.json`` schema file, or a bare name
        (e.g. ``"lif"``) which is resolved against the bundled schemas.

    Returns
    -------
    dict
        Parsed schema dictionary.

    Raises
    ------
    FileNotFoundError
        If the schema file cannot be found.
    ValueError
        If the schema version is unsupported.
    """
    path = Path(source)

    # Bare name resolution: "lif" -> model_schemas/lif.toml or .json
    if not path.suffix:
        for ext in (".toml", ".json"):
            candidate = _SCHEMA_DIR / f"{path.name}{ext}"
            if candidate.exists():
                path = candidate
                break
        else:
            raise FileNotFoundError(f"No schema found for {source!r} in {_SCHEMA_DIR}")
    elif not path.exists():
        raise FileNotFoundError(f"Schema file not found: {path}")

    if path.suffix == ".toml":
        schema = _load_toml(path)
    elif path.suffix == ".json":
        schema = _load_json(path)
    else:
        raise ValueError(f"Unsupported schema format: {path.suffix!r} (use .toml or .json)")

    # Version gate
    version = schema.get("metadata", {}).get("schema_version", 1)
    if version not in _SUPPORTED_SCHEMA_VERSIONS:
        raise ValueError(
            f"Schema version {version} is not supported. Supported: {_SUPPORTED_SCHEMA_VERSIONS}"
        )

    return schema


def schema_to_json(schema: dict[str, Any]) -> str:
    """Export a schema dictionary as a JSON string (for Studio interchange).

    Parameters
    ----------
    schema : dict
        A schema dictionary (as returned by :func:`load_schema`).

    Returns
    -------
    str
        JSON string with 2-space indentation.
    """
    return json.dumps(schema, indent=2, ensure_ascii=False)


def schema_to_toml(schema: dict[str, Any]) -> str:
    """Export a schema dictionary as a TOML string (for version control).

    Parameters
    ----------
    schema : dict
        A schema dictionary.

    Returns
    -------
    str
        TOML-formatted string.
    """
    lines: list[str] = []

    # Ordered sections
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
        data = schema.get(section)
        if not data:
            continue
        lines.append(f"\n[{section}]")
        for key, value in data.items():
            if isinstance(value, str):
                lines.append(f'{key} = "{value}"')
            elif isinstance(value, bool):
                lines.append(f"{key} = {'true' if value else 'false'}")
            elif isinstance(value, (int, float)):
                lines.append(f"{key} = {value}")
            else:
                lines.append(f"{key} = {json.dumps(value)}")

    return "\n".join(lines).strip() + "\n"


class UniversalNeuron:
    """Schema-driven neuron model — parallel meta-layer over existing models.

    Does NOT replace hand-crafted model files.  Delegates simulation to
    :class:`~sc_neurocore.neurons.equation_builder.EquationNeuron`, ensuring
    all existing AST safety, integration methods, and compilation paths
    remain available.

    Parameters
    ----------
    schema : dict
        Parsed schema dictionary (from :func:`load_schema`).
    parameter_overrides : dict, optional
        Override default parameter values (e.g. for parameter sweeps).
    dt_override : float, optional
        Override the schema's default timestep.
    method_override : str, optional
        Override the integration method.
    """

    def __init__(
        self,
        schema: dict[str, Any],
        *,
        parameter_overrides: dict[str, float] | None = None,
        dt_override: float | None = None,
        method_override: str | None = None,
    ) -> None:
        """Initialise from a validated schema dictionary."""
        self._schema = deepcopy(schema)
        self._metadata = self._schema.get("metadata", {})

        # Build parameters
        params = dict(self._schema.get("parameters", {}))
        if parameter_overrides:
            params.update(parameter_overrides)

        # Build initial state
        state = dict(self._schema.get("state", {}))

        # Build dynamics (ODE right-hand sides)
        dynamics = dict(self._schema.get("dynamics", {}))
        if not dynamics:
            raise ValueError("Schema must define at least one ODE in [dynamics]")

        # Integration config
        integration = self._schema.get("integration", {})
        dt = dt_override or integration.get("dt", 0.1)
        method = method_override or integration.get("method", "euler")
        # ``substeps`` (default 1) advances the integrator this many inner steps per
        # macro ``step()`` before a single spike decision, mirroring the conductance
        # hand models' fixed sub-stepping (e.g. 100 dt sub-steps per 1 ms macro step).
        substeps = int(integration.get("substeps", 1))

        # Threshold and reset
        threshold_config = self._schema.get("threshold", {})
        threshold_expr = threshold_config.get("condition")
        detection = threshold_config.get("detection", "level")
        reset_config = self._schema.get("reset", {})

        # Extensions (stored for future use, not consumed by EquationNeuron)
        self._extensions = self._schema.get("extensions", {})
        # schema_version 2 optional layered sections. All default to empty so v1
        # schemas are completely unaffected; these are the authored science/knowledge
        # layers (master plan §4A, invariant I7 — science authored, engineering derived).
        self._science = self._schema.get("science", {})
        self._validation = self._schema.get("validation", {})
        self._provenance = self._schema.get("provenance", {})
        self._hints = self._schema.get("hints", {})

        # Create the underlying EquationNeuron
        self._neuron = EquationNeuron(
            equations=dynamics,
            parameters=params,
            state=state,
            threshold=threshold_expr,
            reset=reset_config if reset_config else None,
            dt=dt,
            method=method,
            detection=detection,
            substeps=substeps,
        )

        logger.debug(
            "UniversalNeuron created: %s (%s, %d state vars, dt=%s, method=%s)",
            self.name,
            self._metadata.get("year", "?"),
            len(state),
            dt,
            method,
        )

    @classmethod
    def from_schema(
        cls,
        source: str | Path,
        **kwargs: Any,
    ) -> UniversalNeuron:
        """Create a UniversalNeuron from a schema file or bundled name.

        Parameters
        ----------
        source : str or Path
            Path to ``.toml``/``.json`` file, or bare name (e.g. ``"lif"``).
        **kwargs
            Forwarded to :class:`UniversalNeuron` constructor.

        Returns
        -------
        UniversalNeuron
            Instantiated neuron model.
        """
        schema = load_schema(source)
        return cls(schema, **kwargs)

    @classmethod
    def from_dict(cls, schema: dict[str, Any], **kwargs: Any) -> UniversalNeuron:
        """Create from an in-memory schema dictionary."""
        return cls(schema, **kwargs)

    # --- Simulation interface ---

    def step(self, **inputs: float) -> int:
        """Advance one timestep.  Keyword args are external inputs (e.g. I=10.0)."""
        return self._neuron.step(**inputs)

    def reset(self) -> None:
        """Reset state to initial conditions."""
        self._neuron.reset()

    @property
    def state(self) -> dict[str, float]:
        """Current state variable values."""
        return self._neuron.state

    @property
    def name(self) -> str:
        """Model name from metadata."""
        return str(self._metadata.get("name", "UnnamedModel"))

    @property
    def doi(self) -> str | None:
        """DOI of the source publication, if available."""
        doi: str | None = self._metadata.get("doi")
        return doi

    @property
    def schema(self) -> dict[str, Any]:
        """Return a deep copy of the underlying schema."""
        return deepcopy(self._schema)

    @property
    def extensions(self) -> dict[str, Any]:
        """Forward-compatible extension fields."""
        return dict(self._extensions)

    @property
    def science(self) -> dict[str, Any]:
        """Authored science layer (schema v2), empty for v1 schemas.

        As-published equations, derivation note, parameter units/sources,
        assumptions, validity range, known limitations and references —
        author-owned and auditable. Engineering artefacts (RTL, polyglot source,
        PPA) are derived elsewhere and never stored here (master plan I7).
        """
        return dict(self._science)

    @property
    def validation(self) -> dict[str, Any]:
        """Authored validation layer (schema v2), empty for v1 schemas.

        Model class, the class-correct validation metric, the cited reference,
        tolerance, the committed evidence artefact and the audit record — what a
        scientific auditor checks.
        """
        return dict(self._validation)

    @property
    def provenance(self) -> dict[str, Any]:
        """Authored provenance layer (schema v2), empty for v1 schemas.

        The per-model changelog and the contributor/credit ledger.
        """
        return dict(self._provenance)

    @property
    def hints(self) -> dict[str, Any]:
        """Optional engineering hints (schema v2), empty for v1 schemas.

        Advisory only (e.g. recommended precision, integrator rationale) — never
        derived artefacts.
        """
        return dict(self._hints)

    # --- Export ---

    def to_json(self) -> str:
        """Export the model schema as JSON (Studio interchange format)."""
        return schema_to_json(self._schema)

    def to_toml(self) -> str:
        """Export the model schema as TOML (version control format)."""
        return schema_to_toml(self._schema)

    def to_equation_neuron(self) -> EquationNeuron:
        """Return the underlying EquationNeuron instance."""
        return self._neuron

    def to_verilog(self, module_name: str | None = None, **kwargs: Any) -> str:
        """Compile the model to Verilog via the equation compiler.

        Parameters
        ----------
        module_name : str, optional
            Verilog module name.  Defaults to a sanitised version of the
            model name.
        **kwargs
            Forwarded to :func:`~sc_neurocore.compiler.equation_compiler.compile_to_verilog`.

        Returns
        -------
        str
            Generated Verilog source.
        """
        from sc_neurocore.compiler.equation_compiler import compile_to_verilog

        if module_name is None:
            # Sanitise model name to valid Verilog identifier
            raw = self.name.lower().replace("-", "_").replace(" ", "_")
            module_name = "sc_" + "".join(c for c in raw if c.isalnum() or c == "_")

        return compile_to_verilog(self._neuron, module_name=module_name, **kwargs)

    # --- Introspection ---

    def list_state_variables(self) -> list[str]:
        """Return the names of all state variables."""
        return list(self._schema.get("state", {}).keys())

    def list_parameters(self) -> list[str]:
        """Return the names of all parameters."""
        return list(self._schema.get("parameters", {}).keys())

    def list_equations(self) -> dict[str, str]:
        """Return the ODE right-hand sides."""
        return dict(self._schema.get("dynamics", {}))

    def __repr__(self) -> str:
        """Return a human-readable summary string."""
        vars_str = ", ".join(self.list_state_variables())
        return f"UniversalNeuron({self.name!r}, state=[{vars_str}])"


def list_bundled_schemas() -> list[str]:
    """Return names of all bundled model schemas (without extensions).

    Returns
    -------
    list of str
        Sorted list of available schema names.
    """
    if not _SCHEMA_DIR.exists():
        return []
    names: set[str] = set()
    for p in _SCHEMA_DIR.iterdir():
        if p.suffix in (".toml", ".json"):
            names.add(p.stem)
    return sorted(names)

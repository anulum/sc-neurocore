# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Ordered authored equation IR requests

"""Preserve authored model order, units and exact random keys across compiler jobs.

Equation and reset declarations are arrays because the process ledger sorts JSON
object keys. Gauss–Seidel integration cannot exchange authored equations without
changing behaviour.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)
from typing_extensions import Self

from sc_neurocore.neurons._stochastic_threshold import DEFAULT_LFSR16_SEED

FiniteNumber = (
    Annotated[int, Field(strict=True)] | Annotated[float, Field(strict=True, allow_inf_nan=False)]
)
NonemptyText = Annotated[str, StringConstraints(strict=True, min_length=1)]
ExactSeed = Annotated[
    str,
    StringConstraints(strict=True, min_length=1, max_length=20, pattern=r"^(0|[1-9][0-9]*)$"),
]
IntegrationMethod = Literal["euler", "rk4", "exp_euler", "map", "gauss_seidel"]


class EquationValue(BaseModel):
    """A finite magnitude and optional authored physical unit.

    Attributes
    ----------
    value : int or float
        Magnitude in the declared unit, or the unannotated numerical convention.
    unit : str or None
        Pint expression whose dimensions are checked by the model constructor.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)
    value: FiniteNumber
    unit: NonemptyText | None = None


class NamedEquationValue(EquationValue):
    """An ordered initial state, parameter or immutable constant.

    Attributes
    ----------
    name : str
        Authored symbol passed unchanged to the equation model.
    """

    name: NonemptyText


class EquationExpression(BaseModel):
    """An ordered state right-hand side or reset expression.

    Attributes
    ----------
    name : str
        State variable updated by the declaration.
    expression : str
        Authored right-hand side under the numerical and event semantics.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)
    name: NonemptyText
    expression: NonemptyText


class AuthoredEquationModel(BaseModel):
    """Serializable constructor inputs of a native-compilable equation model.

    Attributes
    ----------
    equations, initial_state, parameters, constants, reset : list
        Ordered declarations. Repeated names refuse before dictionary conversion.
        An empty initial state selects the public constructor's default.
    dt : EquationValue
        Positive inner timestep, with an explicit time unit in strict mode.
    method : str
        Maintained integrator; order is retained for Gauss–Seidel updates.
    threshold, rate_expression, probability_expression, spike_value : str or None
        Authored event condition, stochastic threshold and graded-spike rules.
    detection, threshold_rng : str
        Maintained event detection and threshold random generator.
    substeps : int
        Positive inner steps per macro timestep.
    rng_seed : str
        Canonical decimal unsigned 64-bit key; strings avoid JavaScript loss.
    random_stream : int
        Unsigned 32-bit counter-based random stream.
    units : str
        Explicit dimensional-validation mode.
    input_unit : EquationValue or None
        Authored quantity declaring the drive's physical dimension.

    Notes
    -----
    A live Python noise_rng object is not a serializable native compiler input.
    No such object is reconstructed or substituted from this request.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)
    equations: Annotated[list[EquationExpression], Field(min_length=1)]
    initial_state: list[NamedEquationValue] = Field(default_factory=list)
    parameters: list[NamedEquationValue] = Field(default_factory=list)
    constants: list[NamedEquationValue] = Field(default_factory=list)
    reset: list[EquationExpression] = Field(default_factory=list)
    threshold: NonemptyText | None = None
    dt: EquationValue = Field(default_factory=lambda: EquationValue(value=0.1))
    method: IntegrationMethod = "euler"
    detection: Literal["level", "crossing", "poisson", "escape_rate"] = "level"
    substeps: Annotated[int, Field(strict=True, ge=1)] = 1
    rate_expression: NonemptyText | None = None
    probability_expression: NonemptyText | None = None
    spike_value: NonemptyText | None = None
    threshold_rng: Literal["lfsr16", "philox"] = "lfsr16"
    rng_seed: ExactSeed = str(DEFAULT_LFSR16_SEED)
    random_stream: Annotated[int, Field(strict=True, ge=0, lt=2**32)] = 0
    units: Literal["none", "strict"] = "none"
    input_unit: EquationValue | None = None

    @field_validator("rng_seed")
    @classmethod
    def _exact_seed(cls, value: str) -> str:
        """Require a canonical decimal key in the native unsigned 64-bit domain."""
        parsed = int(value)
        if parsed >= 2**64 or value != str(parsed):
            raise ValueError("rng_seed must be a canonical unsigned 64-bit decimal string")
        return value

    @model_validator(mode="after")
    def _declarations(self) -> Self:
        """Reject lossy declaration conversion and undeclared state writes."""
        groups = (
            ("equations", [item.name for item in self.equations]),
            ("initial_state", [item.name for item in self.initial_state]),
            ("parameters", [item.name for item in self.parameters]),
            ("constants", [item.name for item in self.constants]),
            ("reset", [item.name for item in self.reset]),
        )
        for label, names in groups:
            if len(names) != len(set(names)):
                raise ValueError(f"{label} cannot repeat a symbol")
        declared = {item.name for item in self.equations}
        if any(item.name not in declared for item in self.initial_state):
            raise ValueError("initial_state contains an undeclared state")
        if any(item.name not in declared for item in self.reset):
            raise ValueError("reset contains an undeclared state")
        if self.dt.value <= 0:
            raise ValueError("dt must have a positive magnitude")
        values = [self.dt, *self.initial_state, *self.parameters, *self.constants]
        if self.input_unit is not None:
            values.append(self.input_unit)
            if self.input_unit.unit is None:
                raise ValueError("input_unit requires a physical unit")
        if self.units == "none" and any(item.unit is not None for item in values):
            raise ValueError("physical unit annotations require units='strict'")
        return self


class EquationIRProfile(BaseModel):
    """Native scalar geometry, product-register and function-lowering choices.

    Attributes
    ----------
    module_name : str
        Authored HDL identifier, validated by the compiler.
    data_width, fraction : int
        Total scalar and fractional bits; the compiler admits geometry.
    signed : bool
        Requested signedness; unsupported datapaths refuse explicitly.
    overflow, rounding : str
        Integrated-state overflow and product rounding semantics.
    pipeline_stages, pipeline_points : int and list of str
        Global product-register selection or explicit product names.
    function_lowering, random_precision : str
        Existing function and normal-draw lowering profiles.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)
    module_name: NonemptyText = "sc_equation_neuron"
    data_width: Annotated[int, Field(strict=True, ge=1)] = 16
    fraction: Annotated[int, Field(strict=True, ge=0)] = 8
    signed: Annotated[bool, Field(strict=True)] = True
    overflow: Literal["saturate", "wrap", "trap"] = "saturate"
    rounding: Literal["truncate", "nearest", "bankers"] = "truncate"
    pipeline_stages: Annotated[int, Field(strict=True, ge=0)] = 0
    pipeline_points: list[NonemptyText] = Field(default_factory=list)
    function_lowering: Literal["lut-v1", "v2"] = "lut-v1"
    random_precision: Literal["standard", "exact"] = "standard"

    @model_validator(mode="after")
    def _product_selection(self) -> Self:
        """Keep global and per-product requests distinct and unambiguous."""
        if self.pipeline_stages and self.pipeline_points:
            raise ValueError("pipeline_stages and pipeline_points are mutually exclusive")
        if len(self.pipeline_points) != len(set(self.pipeline_points)):
            raise ValueError("pipeline_points cannot repeat a product")
        return self


class EquationIRRequest(BaseModel):
    """Versioned authored model and profile for a bounded compiler job.

    Attributes
    ----------
    schema_version : str
        Required descriptive identity; old lossy bodies are not made equivalent
        by silently filling missing source declarations.
    model : AuthoredEquationModel
        Complete serializable model declaration.
    profile : EquationIRProfile
        Numerical and hardware-output choices.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal["studio.equation-ir-request.v1"]
    model: AuthoredEquationModel
    profile: EquationIRProfile = Field(default_factory=EquationIRProfile)


class EquationIRTextRequest(BaseModel):
    """Native graph text for bounded structural or semantic inspection.

    Attributes
    ----------
    ir_text : str
        Actual native text, including its source binding when present.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)
    ir_text: NonemptyText


class EquationIREmitRequest(EquationIRTextRequest):
    """SystemVerilog emission with an explicit source-consistency requirement.

    Attributes
    ----------
    source_policy : str
        Require a replayable source by default. Structural-only emission permits
        unsourced native graphs, without making an authored-model claim.
        A source binding, when present, must always replay successfully.
    """

    source_policy: Literal["require-bound", "structural-only"] = "require-bound"

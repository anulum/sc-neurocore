# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Authored equation native compiler adapter

"""Compile authored models and distinguish graph structure from source replay."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from sc_neurocore.compiler.equation_ir_compiler import compile_to_ir, recompile_ir
from sc_neurocore.exceptions import SCDependencyError
from sc_neurocore.neurons.equation_builder import EquationNeuron
from sc_neurocore.studio.equation_ir_schema import (
    AuthoredEquationModel,
    EquationIREmitRequest,
    EquationIRRequest,
    EquationIRTextRequest,
    EquationValue,
)

if TYPE_CHECKING:
    from sc_neurocore_engine.ir import ScGraph

    from sc_neurocore.compiler.equation_ir_compiler import EquationIR

IRStage = Literal["request", "model", "lowering", "structure", "replay", "emission"]
IROperation = Literal["build", "verify", "replay", "emit-sv"]
IRRefusalCode = Literal[
    "invalid_input", "unsupported_request", "dependency_unavailable", "source_required"
]
T = TypeVar("T")


class EquationIRRefusal(ValueError):
    """A controlled compiler refusal without generated exception diagnostics.

    Attributes
    ----------
    stage : str
        Compiler boundary at which the request refused.
    code : str
        Stable descriptive reason suitable for API and UI handling.
    message : str
        Fixed caller-facing explanation.
    status_code : int
        HTTP input refusal or unavailable dependency status.
    """

    def __init__(self, stage: IRStage, code: IRRefusalCode, message: str) -> None:
        self.stage = stage
        self.code = code
        self.message = message
        self.status_code = 503 if code == "dependency_unavailable" else 422
        super().__init__(message)


class EquationIRPort(BaseModel):
    """A native output's name and physical scalar width.

    Attributes
    ----------
    name : str
        Actual emitted output port.
    width : int
        Physical bit width, including Boolean event outputs.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)
    name: str
    width: int


class EquationIRResult(BaseModel):
    """Native graph result with narrowly scoped validation claims.

    Attributes
    ----------
    operation : str
        Completed compiler action.
    ir_text, graph_name : str
        Canonical native graph and its actual module name.
    n_ops, n_inputs, n_outputs : int
        Actual graph operation and data-port counts; implicit clocks are not inputs.
    valid, errors : bool and list of str
        Structural verification only, never a hardware qualification.
    source_present : bool
        Whether the native graph contains a source payload.
    source_replay : str
        Semantic replay was passed, unavailable, or not requested.
    source_sha256, source_graph_sha256 : str or None
        Native source-binding hashes, without an authentication claim.
    clocks_per_timestep, pipeline_registers : int or None
        Actual compiler output contract, absent on structure-only inspection.
    output_widths : list of EquationIRPort or None
        Ordered physical output widths when a model has been compiled or replayed.
    systemverilog : str or None
        Actual native emitted RTL for the emission operation.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)
    operation: IROperation
    ir_text: str
    graph_name: str
    n_ops: int
    n_inputs: int
    n_outputs: int
    valid: bool
    errors: list[str] = Field(default_factory=list)
    source_present: bool
    source_replay: Literal["passed", "unavailable", "not_requested"]
    source_sha256: str | None
    source_graph_sha256: str | None
    clocks_per_timestep: int | None = None
    pipeline_registers: int | None = None
    output_widths: list[EquationIRPort] | None = None
    systemverilog: str | None = None


def build_equation_ir(request: EquationIRRequest) -> EquationIRResult:
    """Compile the exact ordered source and numeric profile to native IR.

    Parameters
    ----------
    request : EquationIRRequest
        Full serializable constructor inputs and hardware profile.

    Returns
    -------
    EquationIRResult
        Actual graph, source digests, cadence and output contract.

    Raises
    ------
    EquationIRRefusal
        Source, profile or native dependency cannot be admitted.
    """
    neuron = _admit("model", lambda: equation_model(request.model))
    profile = request.profile
    compiled = _admit(
        "lowering",
        lambda: compile_to_ir(
            neuron,
            profile.module_name,
            profile.data_width,
            profile.fraction,
            signed=profile.signed,
            overflow=profile.overflow,
            rounding=profile.rounding,
            pipeline_stages=profile.pipeline_stages,
            pipeline_points=profile.pipeline_points,
            function_lowering=profile.function_lowering,
            random_precision=profile.random_precision,
        ),
    )
    return _result("build", compiled.graph, compiled=compiled)


def equation_model(model: AuthoredEquationModel) -> EquationNeuron:
    """Construct the public model without stepping or replacing its random keys.

    Parameters
    ----------
    model : AuthoredEquationModel
        Ordered declarations, physical quantities and exact random seed.

    Returns
    -------
    EquationNeuron
        Unstepped model with every serializable constructor setting preserved.

    Notes
    -----
    Quantity conversion is confined to Pint's optional untyped boundary. Native
    arithmetic and dimensional validation remain owned by the public constructor.
    """
    return EquationNeuron(
        {item.name: item.expression for item in model.equations},
        parameters={item.name: _quantity(item) for item in model.parameters},
        state={item.name: _quantity(item) for item in model.initial_state} or None,
        threshold=model.threshold,
        reset={item.name: item.expression for item in model.reset},
        constants={item.name: _quantity(item) for item in model.constants},
        dt=_quantity(model.dt),
        method=model.method,
        units=model.units,
        input_unit=None if model.input_unit is None else _quantity(model.input_unit),
        detection=model.detection,
        substeps=model.substeps,
        rate_expression=model.rate_expression,
        probability_expression=model.probability_expression,
        rng_seed=int(model.rng_seed),
        random_stream=model.random_stream,
        spike_value=model.spike_value,
        threshold_rng=model.threshold_rng,
    )


def verify_equation_ir(request: EquationIRTextRequest) -> EquationIRResult:
    """Inspect graph structure without claiming semantic source consistency.

    Parameters
    ----------
    request : EquationIRTextRequest
        Native graph text.

    Returns
    -------
    EquationIRResult
        Structural diagnostics and source availability, without replayed metadata.

    Raises
    ------
    EquationIRRefusal
        Native parsing or dependency admission fails.
    """
    graph = _admit("structure", lambda: _parse(request.ir_text))
    return _result("verify", graph)


def replay_equation_ir(request: EquationIRTextRequest) -> EquationIRResult:
    """Recompile bound source and require equality with the submitted graph.

    Parameters
    ----------
    request : EquationIRTextRequest
        Source-bound native graph text.

    Returns
    -------
    EquationIRResult
        Semantic source consistency and actual reconstructed output contract.

    Raises
    ------
    EquationIRRefusal
        Source is missing, malformed, inconsistent or unsupported.
    """
    graph = _admit("structure", lambda: _parse(request.ir_text))
    _require_source(graph)
    compiled = _admit("replay", lambda: recompile_ir(request.ir_text))
    return _result("replay", compiled.graph, compiled=compiled, replayed=True)


def emit_equation_systemverilog(request: EquationIREmitRequest) -> EquationIRResult:
    """Emit native RTL, enforcing replay whenever source metadata is present.

    Parameters
    ----------
    request : EquationIREmitRequest
        Graph text and explicit source requirement.

    Returns
    -------
    EquationIRResult
        Native SystemVerilog and scoped graph/source validation results.

    Raises
    ------
    EquationIRRefusal
        Structural validity, required source replay or native emission refuses.
    """
    graph = _admit("structure", lambda: _parse(request.ir_text))
    if graph.verify():
        raise EquationIRRefusal("structure", "invalid_input", "Graph structure is invalid.")
    compiled = None
    if graph.source_payload is not None:
        compiled = _admit("replay", lambda: recompile_ir(request.ir_text))
        graph = compiled.graph
    elif request.source_policy == "require-bound":
        _require_source(graph)
    rtl = _admit("emission", graph.emit_sv)
    return _result("emit-sv", graph, compiled=compiled, replayed=compiled is not None, rtl=rtl)


def _quantity(value: EquationValue) -> Any:
    """Preserve exact scalars or translate one quantity at the Pint boundary."""
    if value.unit is None:
        return value.value
    from sc_neurocore.neurons._units import UNIT_REGISTRY, require_pint

    require_pint()
    from pint.errors import PintError

    try:
        return value.value * UNIT_REGISTRY(value.unit)
    except (PintError, SyntaxError) as exc:
        raise ValueError("Invalid physical unit annotation") from exc


def _parse(text: str) -> ScGraph:
    """Load the optional native parser only when a compiler action is requested."""
    from sc_neurocore_engine.ir import parse_ir

    return parse_ir(text)


def _require_source(graph: ScGraph) -> None:
    """Refuse absent source without treating structural validity as replay."""
    if graph.source_payload is None:
        raise EquationIRRefusal(
            "replay", "source_required", "Semantic replay requires bound equation source."
        )


def _admit(stage: IRStage, operation: Callable[[], T]) -> T:
    """Translate expected compiler refusals without exposing interpreter messages."""
    try:
        return operation()
    except (ImportError, SCDependencyError) as exc:
        raise EquationIRRefusal(
            stage, "dependency_unavailable", "A required compiler dependency is unavailable."
        ) from exc
    except NotImplementedError as exc:
        raise EquationIRRefusal(
            stage, "unsupported_request", "The compiler does not support this request."
        ) from exc
    except (ValueError, TypeError, OverflowError) as exc:
        raise EquationIRRefusal(
            stage, "invalid_input", "The authored source or compiler input is invalid."
        ) from exc


def _result(
    operation: IROperation,
    graph: ScGraph,
    *,
    compiled: EquationIR | None = None,
    replayed: bool = False,
    rtl: str | None = None,
) -> EquationIRResult:
    """Read native graph facts and label the specific validation actually run."""
    errors = graph.verify() or []
    present = graph.source_payload is not None
    return EquationIRResult(
        operation=operation,
        ir_text=graph.to_text(),
        graph_name=graph.name,
        n_ops=len(graph),
        n_inputs=graph.num_inputs,
        n_outputs=graph.num_outputs,
        valid=not errors,
        errors=errors,
        source_present=present,
        source_replay="passed" if replayed else "not_requested" if present else "unavailable",
        source_sha256=graph.source_sha256,
        source_graph_sha256=graph.source_graph_sha256,
        clocks_per_timestep=None if compiled is None else compiled.clocks_per_timestep,
        pipeline_registers=None if compiled is None else compiled.pipeline_registers,
        output_widths=None
        if compiled is None
        else [EquationIRPort(name=name, width=width) for name, width in compiled.output_widths],
        systemverilog=rtl,
    )

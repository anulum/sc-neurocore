# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Authored Studio native compiler contracts

"""Real authored models reach the native compiler with source-consistency checks."""

from __future__ import annotations

import json
import unittest
from collections.abc import Callable

from tests.engine_requirement import require_engine

require_engine()

from sc_neurocore_engine.ir import ScGraphBuilder, parse_ir

from sc_neurocore.studio.equation_ir import (
    EquationIRRefusal,
    EquationIRResult,
    build_equation_ir,
    emit_equation_systemverilog,
    equation_model,
    replay_equation_ir,
    verify_equation_ir,
)
from sc_neurocore.studio.equation_ir_schema import (
    AuthoredEquationModel,
    EquationExpression,
    EquationIREmitRequest,
    EquationIRProfile,
    EquationIRRequest,
    EquationIRTextRequest,
    EquationValue,
    NamedEquationValue,
)


def _request(expression: str = "I", initial: int = 0) -> EquationIRRequest:
    """Declare a real single-state model using the public request schema."""
    return EquationIRRequest(
        schema_version="studio.equation-ir-request.v1",
        model=AuthoredEquationModel(
            equations=[EquationExpression(name="v", expression=expression)],
            initial_state=[NamedEquationValue(name="v", value=initial)],
            dt=EquationValue(value=0.5),
        ),
    )


class AuthoredEquationCompilerTests(unittest.TestCase):
    """Exercise the public Studio adapter against the actual native engine."""

    def test_authored_equations_and_initial_state_change_the_graph(self) -> None:
        """Different dynamics and nonzero initial state cannot return fixed LIF."""
        first = build_equation_ir(_request())
        derivative = build_equation_ir(_request("2 * I"))
        initial = build_equation_ir(_request(initial=7))
        self.assertNotEqual(first.ir_text, derivative.ir_text)
        self.assertNotEqual(first.ir_text, initial.ir_text)
        self.assertNotIn("lif_step", first.ir_text)
        self.assertNotIn('sc.input "clk"', first.ir_text)
        self.assertTrue(first.valid)
        self.assertEqual(first.errors, [])
        self.assertEqual(first.source_replay, "not_requested")
        self.assertEqual(first.n_inputs, 1)
        self.assertGreater(first.n_ops, 8)
        self.assertEqual(first.graph_name, "sc_equation_neuron")
        self.assertTrue(first.source_sha256)
        self.assertTrue(first.source_graph_sha256)

    def test_every_maintained_integrator_replays(self) -> None:
        """All five integration choices survive actual source reconstruction."""
        for method in ("euler", "rk4", "exp_euler", "map", "gauss_seidel"):
            with self.subTest(method=method):
                request = _request("I - v")
                model = AuthoredEquationModel.model_validate(
                    {**request.model.model_dump(), "method": method}
                )
                result = build_equation_ir(request.model_copy(update={"model": model}))
                replay = replay_equation_ir(EquationIRTextRequest(ir_text=result.ir_text))
                self.assertEqual(replay.source_replay, "passed")
                self.assertEqual(replay.ir_text, result.ir_text)
                self.assertEqual(replay.output_widths, result.output_widths)

    def test_gauss_seidel_keeps_authored_state_order(self) -> None:
        """Canonical JSON key sorting cannot reorder the equation declaration array."""
        request = _request()
        model = AuthoredEquationModel(
            equations=[
                EquationExpression(name="z", expression="I - a"),
                EquationExpression(name="a", expression="z - a"),
            ],
            method="gauss_seidel",
            initial_state=[
                NamedEquationValue(name="z", value=3),
                NamedEquationValue(name="a", value=1),
            ],
        )
        transported = EquationIRRequest.model_validate_json(
            json.dumps(request.model_copy(update={"model": model}).model_dump(), sort_keys=True)
        )
        result = build_equation_ir(transported)
        source = parse_ir(result.ir_text).source_payload
        self.assertIsNotNone(source)
        payload = json.loads(str(source))
        self.assertEqual([row[0] for row in payload["model"]["equations"]], ["z", "a"])
        reversed_model = model.model_copy(update={"equations": list(reversed(model.equations))})
        reversed_result = build_equation_ir(request.model_copy(update={"model": reversed_model}))
        self.assertNotEqual(result.source_graph_sha256, reversed_result.source_graph_sha256)

    def test_events_constants_parameters_and_exact_random_key(self) -> None:
        """Full declarations, graded reset events and exact random keys are preserved."""
        model = AuthoredEquationModel(
            equations=[EquationExpression(name="v", expression="gain * I + offset")],
            initial_state=[NamedEquationValue(name="v", value=7)],
            parameters=[NamedEquationValue(name="gain", value=2)],
            constants=[NamedEquationValue(name="offset", value=1)],
            threshold="v > 8",
            reset=[EquationExpression(name="v", expression="offset")],
            spike_value="v + gain",
            rng_seed=str(2**64 - 1),
            random_stream=2**32 - 1,
        )
        request = _request().model_copy(update={"model": model})
        result = build_equation_ir(request)
        source = json.loads(str(parse_ir(result.ir_text).source_payload))
        self.assertEqual(source["model"]["rng_seed"], 2**64 - 1)
        self.assertEqual(source["model"]["random_stream"], 2**32 - 1)
        self.assertEqual(source["model"]["constants"], [["offset", 1]])
        self.assertEqual(source["model"]["parameters"], [["gain", 2]])
        self.assertEqual(source["model"]["initial_state"], [["v", 7]])
        self.assertEqual(source["model"]["spike_value"], "v + gain")
        self.assertEqual(source["model"]["reset"], [["v", "offset"]])
        self.assertIsNotNone(result.output_widths)
        self.assertIn("spike_value_out", [port.name for port in result.output_widths or []])
        self.assertGreaterEqual(result.clocks_per_timestep or 0, 1)
        replay_equation_ir(EquationIRTextRequest(ir_text=result.ir_text))

    def test_crossing_substeps_and_unsupported_reset_combination(self) -> None:
        """Supported substeps retain cadence; unsupported reset combinations refuse."""
        request = _request()
        model = request.model.model_copy(
            update={"detection": "crossing", "threshold": "v > 8", "substeps": 2}
        )
        result = build_equation_ir(request.model_copy(update={"model": model}))
        self.assertGreaterEqual(result.clocks_per_timestep or 0, 2)
        self.assertEqual(
            replay_equation_ir(EquationIRTextRequest(ir_text=result.ir_text)).ir_text,
            result.ir_text,
        )
        resetting = model.model_copy(
            update={"reset": [EquationExpression(name="v", expression="0")]}
        )
        with self.assertRaises(EquationIRRefusal) as error:
            build_equation_ir(request.model_copy(update={"model": resetting}))
        self.assertEqual(error.exception.code, "unsupported_request")

    def test_profile_and_actual_registered_output_contract(self) -> None:
        """Actual product-register count, selected Q geometry and cadence are reported."""
        request = _request("gain * v + I")
        model = request.model.model_copy(
            update={"parameters": [NamedEquationValue(name="gain", value=0.5)]}
        )
        profile = EquationIRProfile(
            module_name="authored_cell",
            data_width=32,
            fraction=16,
            rounding="bankers",
            overflow="wrap",
            pipeline_stages=1,
            function_lowering="v2",
            random_precision="exact",
        )
        result = build_equation_ir(request.model_copy(update={"model": model, "profile": profile}))
        self.assertEqual(result.graph_name, "authored_cell")
        self.assertGreater(result.pipeline_registers or 0, 0)
        self.assertGreater(result.clocks_per_timestep or 0, 1)
        self.assertIn(32, [port.width for port in result.output_widths or []])
        source = json.loads(str(parse_ir(result.ir_text).source_payload))
        self.assertEqual(source["numeric"]["fraction"], 16)
        self.assertEqual(source["numeric"]["rounding"], "bankers")
        self.assertEqual(source["numeric"]["overflow"], "wrap")
        replay_equation_ir(EquationIRTextRequest(ir_text=result.ir_text))

    def test_strict_quantities_keep_base_and_display_units(self) -> None:
        """Authored display quantities retain their physical dimension and timestep."""
        model = AuthoredEquationModel(
            equations=[EquationExpression(name="v", expression="(I - v) / tau")],
            initial_state=[NamedEquationValue(name="v", value=-65, unit="millivolt")],
            parameters=[NamedEquationValue(name="tau", value=10, unit="millisecond")],
            dt=EquationValue(value=0.1, unit="millisecond"),
            units="strict",
            input_unit=EquationValue(value=1, unit="millivolt"),
        )
        neuron = equation_model(model)
        self.assertEqual(neuron.units, "strict")
        result = build_equation_ir(
            _request().model_copy(
                update={"model": model, "profile": EquationIRProfile(data_width=32, fraction=16)}
            )
        )
        source = json.loads(str(parse_ir(result.ir_text).source_payload))
        self.assertEqual(source["model"]["display_state_units"]["v"], "millivolt")
        self.assertEqual(
            source["model"]["base_state_units"]["v"], "kilogram * meter ** 2 / ampere / second ** 3"
        )
        self.assertEqual(
            replay_equation_ir(EquationIRTextRequest(ir_text=result.ir_text)).ir_text,
            result.ir_text,
        )

    def test_structure_is_distinct_from_source_replay(self) -> None:
        """Structural inspection reports source availability without claiming replay."""
        result = build_equation_ir(_request())
        inspection = verify_equation_ir(EquationIRTextRequest(ir_text=result.ir_text))
        self.assertTrue(inspection.valid)
        self.assertEqual(inspection.source_replay, "not_requested")
        self.assertIsNone(inspection.output_widths)
        self.assertIsNone(inspection.clocks_per_timestep)
        replay = replay_equation_ir(EquationIRTextRequest(ir_text=result.ir_text))
        self.assertEqual(replay.source_replay, "passed")

    def test_native_emission_replays_bound_source(self) -> None:
        """RTL is emitted through the native source-bound graph path."""
        built = build_equation_ir(_request())
        result = emit_equation_systemverilog(EquationIREmitRequest(ir_text=built.ir_text))
        self.assertIn("module sc_equation_neuron", result.systemverilog or "")
        self.assertEqual(result.source_replay, "passed")
        self.assertEqual(result.output_widths, built.output_widths)

    def test_rehashed_source_binding_cannot_hide_a_different_graph(self) -> None:
        """Native-valid source hashes do not replace actual semantic source replay."""
        built = build_equation_ir(_request())
        payload = parse_ir(built.ir_text).source_payload
        if payload is None:
            self.fail("Compiled authored graph has no source")
        builder = ScGraphBuilder(built.graph_name)
        drive = builder.input("I_t", "i16")
        builder.output("value", drive)
        builder.bind_source(payload)
        substituted = builder.build().to_text()
        inspection = verify_equation_ir(EquationIRTextRequest(ir_text=substituted))
        self.assertTrue(inspection.valid)
        self.assertEqual(inspection.source_replay, "not_requested")
        with self.assertRaises(EquationIRRefusal) as error:
            replay_equation_ir(EquationIRTextRequest(ir_text=substituted))
        self.assertEqual(error.exception.stage, "replay")
        for source_policy in ("require-bound", "structural-only"):
            with self.subTest(source_policy=source_policy), self.assertRaises(EquationIRRefusal):
                emit_equation_systemverilog(
                    EquationIREmitRequest.model_validate(
                        {"ir_text": substituted, "source_policy": source_policy}
                    )
                )

    def test_unknown_unit_refuses_without_generated_diagnostics(self) -> None:
        """An undefined physical unit is an input refusal at the quantity boundary."""
        request = _request()
        model = request.model.model_copy(
            update={"units": "strict", "dt": EquationValue(value=0.1, unit="missing_unit_name")}
        )
        with self.assertRaises(EquationIRRefusal) as error:
            build_equation_ir(request.model_copy(update={"model": model}))
        self.assertEqual(error.exception.code, "invalid_input")
        self.assertEqual(error.exception.stage, "model")

    def test_unsourced_graph_requires_explicit_emission_policy(self) -> None:
        """Raw structural graphs cannot acquire an authored-source verification claim."""
        builder = ScGraphBuilder("raw_cell")
        value = builder.input("i_ext", "fixed<16,8>")
        builder.output("value", value)
        text = builder.build().to_text()
        inspection = verify_equation_ir(EquationIRTextRequest(ir_text=text))
        self.assertTrue(inspection.valid)
        self.assertFalse(inspection.source_present)
        self.assertEqual(inspection.source_replay, "unavailable")
        operations: tuple[Callable[[], EquationIRResult], ...] = (
            lambda: replay_equation_ir(EquationIRTextRequest(ir_text=text)),
            lambda: emit_equation_systemverilog(EquationIREmitRequest(ir_text=text)),
        )
        for operation in operations:
            with self.assertRaises(EquationIRRefusal) as error:
                operation()
            self.assertEqual(error.exception.code, "source_required")
            self.assertEqual(error.exception.status_code, 422)
        emitted = emit_equation_systemverilog(
            EquationIREmitRequest(ir_text=text, source_policy="structural-only")
        )
        self.assertIn("module raw_cell", emitted.systemverilog or "")
        self.assertEqual(emitted.source_replay, "unavailable")
        self.assertIsNone(emitted.pipeline_registers)

    def test_native_parse_refusal_is_controlled(self) -> None:
        """Malformed text exposes an authored structural refusal, not interpreter output."""
        with self.assertRaises(EquationIRRefusal) as error:
            verify_equation_ir(EquationIRTextRequest(ir_text="not a graph"))
        self.assertEqual(error.exception.stage, "structure")
        self.assertEqual(error.exception.code, "invalid_input")
        self.assertEqual(
            error.exception.message, "The authored source or compiler input is invalid."
        )

    def test_structurally_invalid_graph_cannot_emit_rtl(self) -> None:
        """A parsed graph with an undefined operand reports errors and refuses RTL."""
        builder = ScGraphBuilder("invalid_cell")
        builder.input("drive", "i16")
        builder.output("value", 999)
        text = builder.build().to_text()
        inspection = verify_equation_ir(EquationIRTextRequest(ir_text=text))
        self.assertFalse(inspection.valid)
        self.assertTrue(inspection.errors)
        with self.assertRaises(EquationIRRefusal) as error:
            emit_equation_systemverilog(
                EquationIREmitRequest(ir_text=text, source_policy="structural-only")
            )
        self.assertEqual(error.exception.stage, "structure")
        self.assertEqual(error.exception.code, "invalid_input")

    def test_unsupported_width_refuses_without_default_substitution(self) -> None:
        """The native 64-bit port boundary is not silently replaced with Q8.8."""
        request = _request().model_copy(update={"profile": EquationIRProfile(data_width=72)})
        with self.assertRaises(EquationIRRefusal) as error:
            build_equation_ir(request)
        self.assertEqual(error.exception.stage, "lowering")
        self.assertEqual(error.exception.code, "unsupported_request")

    def test_unknown_symbols_refuse_during_lowering(self) -> None:
        """Unresolved symbols cannot become a generic LIF graph."""
        with self.assertRaises(EquationIRRefusal) as error:
            build_equation_ir(_request("unknown_drive + v"))
        self.assertEqual(error.exception.stage, "lowering")
        self.assertEqual(error.exception.code, "invalid_input")

    def test_invalid_model_refuses_before_lowering(self) -> None:
        """Invalid expression syntax refuses at model construction."""
        with self.assertRaises(EquationIRRefusal) as error:
            build_equation_ir(_request("v.__class__"))
        self.assertEqual(error.exception.stage, "model")
        self.assertEqual(error.exception.code, "invalid_input")

# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Authored compiler request admission contracts

"""Reject lossy requests while preserving ordered declarations and exact keys."""

from __future__ import annotations

import json
import unittest

from pydantic import ValidationError

from sc_neurocore.studio.equation_ir_schema import EquationIRRequest


def _body() -> dict[str, object]:
    """Provide one real public ordered model request."""
    return {
        "schema_version": "studio.equation-ir-request.v1",
        "model": {
            "equations": [
                {"name": "z", "expression": "I - z"},
                {"name": "a", "expression": "z - a"},
            ],
            "initial_state": [{"name": "z", "value": 2}, {"name": "a", "value": 1}],
            "rng_seed": str(2**64 - 1),
        },
    }


class EquationIRRequestTests(unittest.TestCase):
    """Exercise the same schema used by HTTP admission and the process worker."""

    def test_sorted_json_transport_keeps_order_and_exact_numbers(self) -> None:
        """Ordered lists and decimal random keys survive the ledger's JSON mode."""
        request = EquationIRRequest.model_validate_json(json.dumps(_body(), sort_keys=True))
        self.assertEqual([item.name for item in request.model.equations], ["z", "a"])
        self.assertEqual([item.name for item in request.model.initial_state], ["z", "a"])
        self.assertEqual(request.model.rng_seed, "18446744073709551615")
        self.assertIs(type(request.model.initial_state[0].value), int)
        self.assertEqual(request.profile.data_width, 16)
        self.assertEqual(request.profile.fraction, 8)
        self.assertEqual(request.profile.function_lowering, "lut-v1")
        round_trip = EquationIRRequest.model_validate_json(request.model_dump_json())
        self.assertEqual(round_trip, request)

    def test_seed_refuses_numeric_loss_and_out_of_domain_text(self) -> None:
        """A key cannot travel through a JavaScript number or noncanonical text."""
        for value in (1, True, 1.5, "-1", "01", " 1", "1e3", str(2**64), ""):
            with self.subTest(value=value), self.assertRaises(ValidationError):
                EquationIRRequest.model_validate(
                    {
                        "schema_version": "studio.equation-ir-request.v1",
                        "model": {
                            "equations": [{"name": "v", "expression": "I"}],
                            "rng_seed": value,
                        },
                    }
                )

    def test_duplicate_declarations_are_not_silently_collapsed(self) -> None:
        """Every ordered declaration category rejects repeated names."""
        for field in ("equations", "reset", "initial_state", "parameters", "constants"):
            declaration = (
                {"name": "v", "expression": "I"}
                if field in ("equations", "reset")
                else {"name": "v", "value": 1}
            )
            model: dict[str, object] = {"equations": [{"name": "v", "expression": "I"}]}
            model[field] = [declaration, declaration]
            with self.subTest(field=field), self.assertRaises(ValidationError):
                EquationIRRequest.model_validate(
                    {"schema_version": "studio.equation-ir-request.v1", "model": model}
                )

    def test_reset_writes_require_an_equation_or_initial_state_declaration(self) -> None:
        """A reset cannot introduce a new state that has neither equation nor initial value."""
        with self.assertRaisesRegex(ValidationError, "reset contains an undeclared state"):
            EquationIRRequest.model_validate(
                {
                    "schema_version": "studio.equation-ir-request.v1",
                    "model": {
                        "equations": [{"name": "v", "expression": "I"}],
                        "reset": [{"name": "missing", "expression": "0"}],
                    },
                }
            )

    def test_partial_numeric_initial_state_keeps_supplied_values_and_implicit_defaults(
        self,
    ) -> None:
        """Transport does not fill omitted values or erase a supplied nonzero state."""
        request = EquationIRRequest.model_validate(
            {
                "schema_version": "studio.equation-ir-request.v1",
                "model": {
                    "equations": [
                        {"name": "v", "expression": "-v + w + I"},
                        {"name": "w", "expression": "-w"},
                    ],
                    "initial_state": [{"name": "v", "value": 2}],
                },
            }
        )
        self.assertEqual(
            [(item.name, item.value) for item in request.model.initial_state], [("v", 2)]
        )
        self.assertEqual([item.name for item in request.model.equations], ["v", "w"])

    def test_numeric_partial_and_held_states_preserve_the_authored_source(self) -> None:
        """Defaulting belongs to the constructor; transport retains the original declaration."""
        body = {
            "schema_version": "studio.equation-ir-request.v1",
            "model": {
                "equations": [{"name": "v", "expression": "I**1.5+clip(I,.25,1)**clip(w,-1,1)"}],
                "initial_state": [{"name": "w", "value": 0.5}],
                "method": "map",
                "dt": {"value": 1},
                "reset": [{"name": "w", "expression": "w + .25"}],
            },
        }
        request = EquationIRRequest.model_validate(body)
        self.assertEqual([item.name for item in request.model.initial_state], ["w"])
        self.assertEqual(request.model.initial_state[0].value, 0.5)
        self.assertEqual(request.model.reset[0].name, "w")
        self.assertEqual(EquationIRRequest.model_validate_json(request.model_dump_json()), request)

    def test_strict_initial_state_cannot_hide_a_missing_equation_behind_a_held_word(self) -> None:
        """Equal array lengths do not prove that every dimensional state has a quantity."""
        with self.assertRaisesRegex(ValidationError, "every equation state"):
            EquationIRRequest.model_validate(
                {
                    "schema_version": "studio.equation-ir-request.v1",
                    "model": {
                        "equations": [{"name": "v", "expression": "I"}],
                        "initial_state": [{"name": "w", "value": 1, "unit": "volt"}],
                        "units": "strict",
                        "dt": {"value": 1, "unit": "second"},
                    },
                }
            )

    def test_scalar_types_and_timestep_are_explicit(self) -> None:
        """Booleans, nonfinite values and nonpositive timesteps cannot be coerced."""
        for value in (True, "0.1", float("nan"), float("inf"), 0, -1):
            with self.subTest(value=value), self.assertRaises(ValidationError):
                EquationIRRequest.model_validate(
                    {
                        "schema_version": "studio.equation-ir-request.v1",
                        "model": {
                            "equations": [{"name": "v", "expression": "I"}],
                            "dt": {"value": value},
                        },
                    }
                )

    def test_unit_annotations_require_strict_mode_and_input_dimension(self) -> None:
        """Physical quantities cannot be silently stripped in the unannotated mode."""
        for addition in (
            {"dt": {"value": 0.1, "unit": "millisecond"}},
            {"input_unit": {"value": 1}},
            {"input_unit": {"value": 1, "unit": "ampere"}},
        ):
            with self.subTest(addition=addition), self.assertRaises(ValidationError):
                EquationIRRequest.model_validate(
                    {
                        "schema_version": "studio.equation-ir-request.v1",
                        "model": {
                            "equations": [{"name": "v", "expression": "I"}],
                            **addition,
                        },
                    }
                )

    def test_product_selection_is_unambiguous(self) -> None:
        """Global and explicit product choices cannot conflict or duplicate."""
        for profile in (
            {"pipeline_stages": 1, "pipeline_points": ["mul_0"]},
            {"pipeline_points": ["mul_0", "mul_0"]},
            {"pipeline_stages": True},
            {"signed": 1},
            {"data_width": 16.0},
        ):
            with self.subTest(profile=profile), self.assertRaises(ValidationError):
                EquationIRRequest.model_validate({**_body(), "profile": profile})

    def test_obsolete_and_unknown_source_fields_refuse(self) -> None:
        """Legacy lossy bodies and custom live RNG objects are not silently upgraded."""
        for body in (
            {"equations": ["dv/dt = I"], "init": {"v": 7}},
            {**_body(), "unknown": True},
            {
                "schema_version": "studio.equation-ir-request.v1",
                "model": {
                    "equations": [{"name": "v", "expression": "I"}],
                    "noise_rng": "replace-me",
                },
            },
            {**_body(), "schema_version": "studio.equation-ir-request.v0"},
        ):
            with self.subTest(body=body), self.assertRaises(ValidationError):
                EquationIRRequest.model_validate(body)

    def test_substeps_stream_and_modes_are_not_coerced(self) -> None:
        """Only supported integration modes and exact bounded integer controls admit."""
        for addition in (
            {"substeps": 0},
            {"substeps": True},
            {"random_stream": 2**32},
            {"random_stream": -1},
            {"method": "adaptive"},
            {"threshold_rng": "system"},
        ):
            with self.subTest(addition=addition), self.assertRaises(ValidationError):
                EquationIRRequest.model_validate(
                    {
                        "schema_version": "studio.equation-ir-request.v1",
                        "model": {
                            "equations": [{"name": "v", "expression": "I"}],
                            **addition,
                        },
                    }
                )

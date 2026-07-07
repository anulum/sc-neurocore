# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for formal-proof transform dispatch

"""Regression coverage for the opt-in compiler proof-transform registry."""

from __future__ import annotations

from typing import cast

import pytest

import sc_neurocore.compiler as compiler
from sc_neurocore.compiler.equivalence_miter import parse_module_interface
from sc_neurocore.compiler.operator_abstraction import LiftedSignal, abstract_to_free_inputs
from sc_neurocore.compiler.proof_transforms import (
    PROOF_TRANSFORMS,
    ProofTransformKind,
    apply_proof_transform,
    get_proof_transform,
    list_proof_transforms,
)
from sc_neurocore.compiler.whitebox_taps import StateTap, expose_state_taps

_MODULE = """`timescale 1ns/1ps
module foo #(parameter integer W = 8)(
    input wire clk,
    input wire signed [W-1:0] a,
    input wire signed [W-1:0] b,
    output reg signed [W-1:0] y
);
    reg signed [W-1:0] state;
    wire signed [2*W-1:0] prod;
    wire signed [W-1:0] scaled;
    assign prod = a * b;
    assign scaled = prod >>> 2;
    always @(posedge clk) begin state <= scaled; y <= state; end
endmodule
"""


def test_package_facade_exposes_registry_symbols() -> None:
    """The compiler package root exposes the proof-transform dispatch facade."""
    assert compiler.PROOF_TRANSFORMS is PROOF_TRANSFORMS
    assert compiler.apply_proof_transform is apply_proof_transform
    assert compiler.get_proof_transform is get_proof_transform
    assert compiler.list_proof_transforms is list_proof_transforms
    assert "apply_proof_transform" in compiler.__all__
    assert "list_proof_transforms" in compiler.__all__


def test_registry_classifies_only_opt_in_transforms() -> None:
    """Registered proof transforms are discoverable and disabled by default."""
    transforms = list_proof_transforms()
    assert transforms is PROOF_TRANSFORMS
    assert {transform.kind for transform in transforms} == {
        "whitebox_taps",
        "operator_abstraction",
    }
    assert all(not transform.default_enabled for transform in transforms)


def test_get_proof_transform_returns_metadata() -> None:
    """Lookup returns the selected transform and rejects unknown names."""
    whitebox = get_proof_transform("whitebox_taps")
    assert whitebox.module == "sc_neurocore.compiler.whitebox_taps"
    assert whitebox.entrypoint == "expose_state_taps"

    with pytest.raises(KeyError, match="unknown proof transform"):
        get_proof_transform("not_registered")


def test_apply_whitebox_transform_dispatches_to_real_implementation() -> None:
    """The registry dispatch path exposes real state taps, not a stub."""
    transformed = apply_proof_transform(
        "whitebox_taps",
        _MODULE,
        top="foo",
        taps=[StateTap("state_tap", "state", msb="W-1", signed=True)],
    )

    assert transformed == expose_state_taps(
        _MODULE,
        top="foo",
        taps=[StateTap("state_tap", "state", msb="W-1", signed=True)],
    )
    ports = parse_module_interface(transformed, "foo", params={"W": 8})
    assert any(port.name == "state_tap" and port.direction == "output" for port in ports)
    assert "assign state_tap = state;" in transformed


def test_apply_operator_abstraction_dispatches_to_real_implementation() -> None:
    """The registry dispatch path lifts multiplier products through the real transform."""
    transformed = apply_proof_transform(
        "operator_abstraction",
        _MODULE,
        top="foo",
        signals=[LiftedSignal("prod", "prod_free", msb="2*W-1", signed=True)],
    )

    assert transformed == abstract_to_free_inputs(
        _MODULE,
        top="foo",
        signals=[LiftedSignal("prod", "prod_free", msb="2*W-1", signed=True)],
    )
    ports = parse_module_interface(transformed, "foo", params={"W": 8})
    assert any(port.name == "prod_free" and port.direction == "input" for port in ports)
    assert "a * b" not in transformed
    assert "assign scaled = prod_free >>> 2;" in transformed


def test_apply_proof_transform_requires_payloads() -> None:
    """Dispatch fails closed when a transform-specific payload is missing."""
    with pytest.raises(ValueError, match="requires taps"):
        apply_proof_transform("whitebox_taps", _MODULE, top="foo")
    with pytest.raises(ValueError, match="requires signals"):
        apply_proof_transform("operator_abstraction", _MODULE, top="foo")


def test_apply_proof_transform_rejects_unknown_kind_at_runtime() -> None:
    """Runtime strings outside the typed transform set fail closed."""
    unknown = cast(ProofTransformKind, "not_registered")
    with pytest.raises(KeyError, match="unknown proof transform"):
        apply_proof_transform(unknown, _MODULE, top="foo")

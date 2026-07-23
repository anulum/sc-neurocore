# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestCheckIRTypes from former test_ir_type_checker.py

"""Focused suite: TestCheckIRTypes from former test_ir_type_checker.py."""

from __future__ import annotations

from tests.ir_type_checker_support import *  # noqa: F403

class TestCheckIRTypes:
    """Coverage for IR graph validation and fail-closed diagnostics."""

    def test_valid_bitstream_graph(self) -> None:
        nodes = {
            "enc": IRNode("enc", "encoder", [], SignalType.BITSTREAM),
            "and": IRNode(
                "and", "and", [SignalType.BITSTREAM, SignalType.BITSTREAM], SignalType.BITSTREAM
            ),
        }
        edges = [IREdge("enc", "and", dst_port=0)]
        errors = check_ir_types(nodes, edges)
        assert len(errors) == 0

    def test_rate_to_bitstream_error(self) -> None:
        nodes = {
            "input": IRNode("input", "input", [], SignalType.RATE),
            "and": IRNode("and", "and", [SignalType.BITSTREAM], SignalType.BITSTREAM),
        }
        edges = [IREdge("input", "and", dst_port=0)]
        errors = check_ir_types(nodes, edges)
        assert len(errors) == 1
        assert "Type mismatch" in errors[0].message
        assert "encoder/decoder" in errors[0].message

    def test_missing_source_node(self) -> None:
        nodes = {"b": IRNode("b", "and", [SignalType.BITSTREAM], SignalType.BITSTREAM)}
        edges = [IREdge("a", "b")]
        errors = check_ir_types(nodes, edges)
        assert len(errors) == 1
        assert "not found" in errors[0].message

    def test_missing_destination_node(self) -> None:
        nodes = {"a": IRNode("a", "enc", [], SignalType.BITSTREAM)}
        edges = [IREdge("a", "z")]
        errors = check_ir_types(nodes, edges)
        assert len(errors) == 1
        assert "not found" in errors[0].message

    def test_port_out_of_range(self) -> None:
        nodes = {
            "a": IRNode("a", "enc", [], SignalType.BITSTREAM),
            "b": IRNode("b", "and", [SignalType.BITSTREAM], SignalType.BITSTREAM),
        }
        edges = [IREdge("a", "b", dst_port=5)]
        errors = check_ir_types(nodes, edges)
        assert len(errors) == 1
        assert "out of range" in errors[0].message

    def test_negative_port_out_of_range(self) -> None:
        nodes = {
            "a": IRNode("a", "enc", [], SignalType.BITSTREAM),
            "b": IRNode("b", "and", [SignalType.BITSTREAM], SignalType.BITSTREAM),
        }
        edges = [IREdge("a", "b", dst_port=-1)]
        errors = check_ir_types(nodes, edges)
        assert len(errors) == 1
        assert "out of range" in errors[0].message

    def test_multiple_errors(self) -> None:
        nodes = {
            "rate_in": IRNode("rate_in", "input", [], SignalType.RATE),
            "fixed_in": IRNode("fixed_in", "input", [], SignalType.FIXED),
            "and": IRNode(
                "and",
                "and",
                [SignalType.BITSTREAM, SignalType.BITSTREAM],
                SignalType.BITSTREAM,
            ),
        }
        edges = [
            IREdge("rate_in", "and", dst_port=0),
            IREdge("fixed_in", "and", dst_port=1),
        ]
        errors = check_ir_types(nodes, edges)
        assert len(errors) == 2

    def test_empty_graph_no_errors(self) -> None:
        errors = check_ir_types({}, [])
        assert len(errors) == 0

    def test_any_type_passes(self) -> None:
        nodes = {
            "a": IRNode("a", "passthrough", [], SignalType.ANY),
            "b": IRNode("b", "sink", [SignalType.BITSTREAM], SignalType.BITSTREAM),
        }
        edges = [IREdge("a", "b", dst_port=0)]
        errors = check_ir_types(nodes, edges)
        assert len(errors) == 0

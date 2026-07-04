# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for IR type checker

"""Tests for Stochastic IR type checking."""

from sc_neurocore.compiler.ir_type_checker import (
    IREdge,
    IRNode,
    SignalType,
    check_ir_types,
    types_compatible,
)


class TestTypesCompatible:
    """Coverage for direct and wildcard signal-type compatibility."""

    def test_same_type_compatible(self) -> None:
        assert types_compatible(SignalType.BITSTREAM, SignalType.BITSTREAM)
        assert types_compatible(SignalType.RATE, SignalType.RATE)
        assert types_compatible(SignalType.SPIKE, SignalType.SPIKE)

    def test_rate_to_bitstream_incompatible(self) -> None:
        assert not types_compatible(SignalType.RATE, SignalType.BITSTREAM)

    def test_bitstream_to_rate_incompatible(self) -> None:
        assert not types_compatible(SignalType.BITSTREAM, SignalType.RATE)

    def test_spike_to_bitstream_compatible(self) -> None:
        assert types_compatible(SignalType.SPIKE, SignalType.BITSTREAM)

    def test_any_matches_everything(self) -> None:
        assert types_compatible(SignalType.ANY, SignalType.BITSTREAM)
        assert types_compatible(SignalType.RATE, SignalType.ANY)


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

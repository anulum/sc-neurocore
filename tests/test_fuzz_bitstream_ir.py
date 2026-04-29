# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Property-based fuzz tests for bitstream and IR boundaries

"""Property-based fuzz tests for bitstream and IR boundary handling."""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from sc_neurocore.accel.vector_ops import pack_bitstream, unpack_bitstream
from sc_neurocore.compiler.ir_type_checker import IREdge, IRNode, SignalType, check_ir_types


@st.composite
def _bit_vector(draw: st.DrawFn, max_size: int = 257) -> np.ndarray:
    length = draw(st.integers(min_value=0, max_value=max_size))
    bits = draw(st.lists(st.integers(min_value=0, max_value=1), min_size=length, max_size=length))
    return np.asarray(bits, dtype=np.uint8)


@st.composite
def _bit_matrix(draw: st.DrawFn) -> np.ndarray:
    rows = draw(st.integers(min_value=1, max_value=8))
    cols = draw(st.integers(min_value=0, max_value=257))
    bits = draw(
        st.lists(st.integers(min_value=0, max_value=1), min_size=rows * cols, max_size=rows * cols)
    )
    return np.asarray(bits, dtype=np.uint8).reshape(rows, cols)


@given(bits=_bit_vector())
@settings(max_examples=100, deadline=None)
def test_fuzz_pack_unpack_1d_roundtrip(bits: np.ndarray) -> None:
    packed = pack_bitstream(bits)
    unpacked = unpack_bitstream(packed, bits.size)
    assert unpacked.dtype == np.uint8
    assert np.array_equal(unpacked, bits)


@given(bits=_bit_matrix())
@settings(max_examples=100, deadline=None)
def test_fuzz_pack_unpack_2d_roundtrip(bits: np.ndarray) -> None:
    packed = pack_bitstream(bits)
    unpacked = unpack_bitstream(packed, bits.size, original_shape=bits.shape)
    assert unpacked.dtype == np.uint8
    assert unpacked.shape == bits.shape
    assert np.array_equal(unpacked, bits)


@given(
    src_type=st.sampled_from(list(SignalType)),
    dst_type=st.sampled_from(list(SignalType)),
    dst_port=st.integers(min_value=-32, max_value=-1),
)
@settings(max_examples=80)
def test_fuzz_ir_rejects_negative_destination_ports(
    src_type: SignalType,
    dst_type: SignalType,
    dst_port: int,
) -> None:
    nodes = {
        "src": IRNode("src", "source", [], src_type),
        "dst": IRNode("dst", "sink", [dst_type], dst_type),
    }
    errors = check_ir_types(nodes, [IREdge("src", "dst", dst_port=dst_port)])
    assert len(errors) == 1
    assert "out of range" in errors[0].message

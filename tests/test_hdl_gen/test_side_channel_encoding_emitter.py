# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for side-channel protected encoding HDL hooks

from __future__ import annotations

import pytest

from sc_neurocore.hdl_gen import SideChannelEncodingEmitter
from sc_neurocore.security import (
    ThermalSCEncodingConfig,
    encode_activity_balanced_probability,
)


def test_side_channel_encoding_emitter_generates_payload_and_dummy_rom_wrapper() -> None:
    record = encode_activity_balanced_probability(
        0.25,
        ThermalSCEncodingConfig(
            bitstream_length=8,
            seed=5,
            dummy_streams_per_record=2,
            max_dummy_overhead_ratio=2.0,
        ),
    )

    verilog = SideChannelEncodingEmitter(
        module_name="protected_side_channel_source",
        encoding=record,
    ).generate()

    assert "module protected_side_channel_source" in verilog
    assert "localparam integer BITSTREAM_LENGTH = 8;" in verilog
    assert "localparam integer DUMMY_STREAMS = 2;" in verilog
    assert "localparam [7:0] PAYLOAD_BITS = 8'b" in verilog
    assert "localparam [15:0] DUMMY_BITS = 16'b" in verilog
    assert "assign payload_bit = PAYLOAD_BITS[sample_index];" in verilog
    assert "assign dummy_bits[0] = DUMMY_BITS[sample_index];" in verilog
    assert "assign dummy_bits[1] = DUMMY_BITS[BITSTREAM_LENGTH + sample_index];" in verilog
    assert "analytic_simulation_only" in verilog
    assert "DPA-resistant" not in verilog


def test_side_channel_encoding_emitter_manifest_links_hdl_to_encoding_evidence() -> None:
    record = encode_activity_balanced_probability(
        0.5,
        ThermalSCEncodingConfig(bitstream_length=8),
    )

    manifest = SideChannelEncodingEmitter(
        module_name="sc_side_channel_hook",
        encoding=record,
    ).manifest(verilog_path="rtl/sc_side_channel_hook.v")

    assert manifest == {
        "schema_version": "sc-neurocore.side-channel-hdl-hook.v0.1",
        "module_name": "sc_side_channel_hook",
        "verilog_path": "rtl/sc_side_channel_hook.v",
        "evidence_boundary": "analytic_simulation_only",
        "bitstream_length": 8,
        "dummy_streams": 0,
        "payload_transitions": 7,
        "dummy_transitions": [],
    }


def test_side_channel_encoding_emitter_rejects_empty_module_name() -> None:
    record = encode_activity_balanced_probability(
        0.5,
        ThermalSCEncodingConfig(bitstream_length=8),
    )

    with pytest.raises(ValueError):
        SideChannelEncodingEmitter(module_name="", encoding=record).generate()


def test_side_channel_encoding_emitter_ties_dummy_bits_low_when_no_streams() -> None:
    record = encode_activity_balanced_probability(
        0.5,
        ThermalSCEncodingConfig(bitstream_length=8, dummy_streams_per_record=0),
    )

    verilog = SideChannelEncodingEmitter(
        module_name="unpadded_source",
        encoding=record,
    ).generate()

    assert "assign dummy_bits = 1'b0;" in verilog


def test_side_channel_encoding_emitter_offsets_third_and_later_dummy_streams() -> None:
    record = encode_activity_balanced_probability(
        0.5,
        ThermalSCEncodingConfig(
            bitstream_length=8,
            seed=5,
            dummy_streams_per_record=3,
            max_dummy_overhead_ratio=4.0,
        ),
    )

    verilog = SideChannelEncodingEmitter(
        module_name="triple_dummy_source",
        encoding=record,
    ).generate()

    assert "assign dummy_bits[2] = DUMMY_BITS[BITSTREAM_LENGTH * 2 + sample_index];" in verilog


def test_bits_literal_renders_empty_tuple_as_single_zero() -> None:
    from sc_neurocore.hdl_gen.side_channel_encoding_emitter import _bits_literal

    assert _bits_literal(()) == "0"

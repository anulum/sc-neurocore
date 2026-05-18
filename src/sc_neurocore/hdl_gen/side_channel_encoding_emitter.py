# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — HDL hook emitter for analytic side-channel encodings

"""Emit HDL hooks for precomputed activity-shaped SC bitstreams."""

from __future__ import annotations

from typing import Any

from sc_neurocore.security import ActivityBalancedEncoding

from ._ident import sanitize_ident

SIDE_CHANNEL_HDL_HOOK_SCHEMA_VERSION = "sc-neurocore.side-channel-hdl-hook.v0.1"


class SideChannelEncodingEmitter:
    """Emit a synthesisable ROM-style wrapper for one protected encoding record."""

    def __init__(
        self,
        *,
        module_name: str = "sc_side_channel_encoding_source",
        encoding: ActivityBalancedEncoding,
    ) -> None:
        self.module_name = sanitize_ident(module_name, context="module name")
        self.encoding = encoding

    def generate(self) -> str:
        """Return a Verilog module exposing payload and dummy stream bits."""

        bitstream_length = len(self.encoding.bitstream)
        dummy_streams = len(self.encoding.dummy_bitstreams)
        payload_bits = _bits_literal(self.encoding.bitstream)
        dummy_bits = _bits_literal(
            tuple(bit for stream in self.encoding.dummy_bitstreams for bit in stream)
        )
        dummy_width = max(dummy_streams, 1)
        sample_width = max((bitstream_length - 1).bit_length(), 1)
        lines = [
            f"module {self.module_name} (",
            f"    input wire [{sample_width - 1}:0] sample_index,",
            "    output wire payload_bit,",
            f"    output wire [{dummy_width - 1}:0] dummy_bits",
            ");",
            "",
            "    // Evidence boundary: analytic_simulation_only.",
            f"    localparam integer BITSTREAM_LENGTH = {bitstream_length};",
            f"    localparam integer DUMMY_STREAMS = {dummy_streams};",
            f"    localparam [{bitstream_length - 1}:0] PAYLOAD_BITS = "
            f"{bitstream_length}'b{payload_bits};",
            f"    localparam [{max(bitstream_length * dummy_streams, 1) - 1}:0] "
            f"DUMMY_BITS = {max(bitstream_length * dummy_streams, 1)}'b{dummy_bits};",
            "",
            "    assign payload_bit = PAYLOAD_BITS[sample_index];",
        ]
        if dummy_streams == 0:
            lines.append("    assign dummy_bits = 1'b0;")
        else:
            for index in range(dummy_streams):
                if index == 0:
                    offset = "sample_index"
                elif index == 1:
                    offset = "BITSTREAM_LENGTH + sample_index"
                else:
                    offset = f"BITSTREAM_LENGTH * {index} + sample_index"
                lines.append(f"    assign dummy_bits[{index}] = DUMMY_BITS[{offset}];")
        lines.append("endmodule")
        return "\n".join(lines)

    def manifest(self, *, verilog_path: str) -> dict[str, Any]:
        """Return transport metadata linking the HDL hook to analytic evidence."""

        return {
            "schema_version": SIDE_CHANNEL_HDL_HOOK_SCHEMA_VERSION,
            "module_name": self.module_name,
            "verilog_path": verilog_path,
            "evidence_boundary": self.encoding.evidence_boundary,
            "bitstream_length": len(self.encoding.bitstream),
            "dummy_streams": len(self.encoding.dummy_bitstreams),
            "payload_transitions": self.encoding.activity_summary.per_stream_transition_counts[0],
            "dummy_transitions": list(
                self.encoding.activity_summary.per_stream_transition_counts[1:]
            ),
        }


def _bits_literal(bits: tuple[int, ...]) -> str:
    if not bits:
        return "0"
    return "".join(str(bit) for bit in reversed(bits))

# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — IR type checker for SC compilation pipeline

"""Type checker for Stochastic IR: catches Bitstream/Rate/Spike mismatches.

Before emitting Verilog or MLIR, the IR graph should be type-checked
to ensure connected nodes have compatible signal types. Without this,
type errors only surface at synthesis time (or worse, produce silent
wrong-answer bugs).

Signal types:
- Bitstream: temporal sequence of {0,1}, encodes probability via density
- Rate: scalar probability in [0,1], no temporal structure
- Spike: binary event (0 or 1), single timestep
- Fixed: Q-format fixed-point integer

Compatible connections:
- Bitstream → Bitstream (native SC)
- Rate → Rate (probability domain)
- Spike → Spike (spiking domain)
- Rate → Bitstream (requires encoder)
- Bitstream → Rate (requires decoder/popcount)
- Spike → Bitstream (direct embedding)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto


class SignalType(Enum):
    """Signal domains accepted by the stochastic IR compatibility checker."""

    BITSTREAM = auto()
    RATE = auto()
    SPIKE = auto()
    FIXED = auto()
    ANY = auto()  # wildcard, matches everything


# Which pairs are directly compatible (no conversion needed)
_COMPATIBLE = {
    (SignalType.BITSTREAM, SignalType.BITSTREAM),
    (SignalType.RATE, SignalType.RATE),
    (SignalType.SPIKE, SignalType.SPIKE),
    (SignalType.FIXED, SignalType.FIXED),
    (SignalType.SPIKE, SignalType.BITSTREAM),  # spike embeds as single-bit stream
    (SignalType.ANY, SignalType.ANY),
}


def types_compatible(src: SignalType, dst: SignalType) -> bool:
    """Check if src can connect to dst without explicit conversion."""
    if src == SignalType.ANY or dst == SignalType.ANY:
        return True
    return (src, dst) in _COMPATIBLE


@dataclass
class IRNode:
    """A typed node in the Stochastic IR graph."""

    name: str
    op: str  # e.g. "and", "mux", "xor", "encoder", "decoder", "lif", "popcount"
    input_types: list[SignalType] = field(default_factory=list)
    output_type: SignalType = SignalType.BITSTREAM


@dataclass
class IREdge:
    """Connection record from one typed IR node port to another."""

    src: str
    dst: str
    src_port: int = 0
    dst_port: int = 0


@dataclass
class IRTypeError:
    """A type mismatch found during checking."""

    src_node: str
    dst_node: str
    src_type: SignalType
    dst_type: SignalType
    message: str


def check_ir_types(
    nodes: dict[str, IRNode],
    edges: list[IREdge],
) -> list[IRTypeError]:
    """Type-check an IR graph and return all type errors.

    Parameters
    ----------
    nodes : dict mapping node name → IRNode
    edges : list of IREdge connections

    Returns
    -------
    list of IRTypeError (empty if all types check out)
    """
    errors: list[IRTypeError] = []

    for edge in edges:
        if edge.src not in nodes:
            errors.append(
                IRTypeError(
                    edge.src,
                    edge.dst,
                    SignalType.ANY,
                    SignalType.ANY,
                    f"Source node '{edge.src}' not found in graph",
                )
            )
            continue
        if edge.dst not in nodes:
            errors.append(
                IRTypeError(
                    edge.src,
                    edge.dst,
                    SignalType.ANY,
                    SignalType.ANY,
                    f"Destination node '{edge.dst}' not found in graph",
                )
            )
            continue

        src_node = nodes[edge.src]
        dst_node = nodes[edge.dst]
        src_type = src_node.output_type

        if edge.dst_port < 0 or edge.dst_port >= len(dst_node.input_types):
            errors.append(
                IRTypeError(
                    edge.src,
                    edge.dst,
                    src_type,
                    SignalType.ANY,
                    f"Port {edge.dst_port} out of range for '{edge.dst}' "
                    f"(has {len(dst_node.input_types)} inputs)",
                )
            )
            continue

        dst_type = dst_node.input_types[edge.dst_port]

        if not types_compatible(src_type, dst_type):
            errors.append(
                IRTypeError(
                    edge.src,
                    edge.dst,
                    src_type,
                    dst_type,
                    f"Type mismatch: {edge.src} outputs {src_type.name} "
                    f"but {edge.dst} port {edge.dst_port} expects {dst_type.name}. "
                    f"Insert a converter (encoder/decoder).",
                )
            )

    return errors

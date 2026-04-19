# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Weight Serialization (ported from tinysc_riscv/weights.rs)

"""Zero-copy weight loading for SC networks.

Binary format for pre-trained SC network weights that can be loaded
from flash/disk without heap allocation. Compatible with the Rust
bare-metal implementation.

Wire format (little-endian):
    [4B magic 0x5343574C] [4B version] [4B n_layers] [4B flags]
    For each layer:
        [4B n_inputs] [4B n_outputs] [4B threshold] [4B reserved]
        [n_outputs × n_words × 4B weight words]
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

WEIGHT_MAGIC = 0x5343_574C  # "SCWL" in LE
WEIGHT_VERSION = 1


@dataclass
class WeightHeader:
    """Weight blob header (16 bytes)."""

    magic: int = WEIGHT_MAGIC
    version: int = WEIGHT_VERSION
    n_layers: int = 0
    flags: int = 0

    def to_bytes(self) -> bytes:
        return struct.pack("<IIII", self.magic, self.version, self.n_layers, self.flags)

    @classmethod
    def from_bytes(cls, data: bytes) -> WeightHeader:
        m, v, nl, f = struct.unpack("<IIII", data[:16])
        return cls(magic=m, version=v, n_layers=nl, flags=f)

    def validate(self) -> bool:
        return self.magic == WEIGHT_MAGIC and self.version <= WEIGHT_VERSION


@dataclass
class LayerHeader:
    """Per-layer header (16 bytes)."""

    n_inputs: int = 0
    n_outputs: int = 0
    threshold: int = 512
    reserved: int = 0

    def to_bytes(self) -> bytes:
        return struct.pack("<IIII", self.n_inputs, self.n_outputs, self.threshold, self.reserved)

    @classmethod
    def from_bytes(cls, data: bytes) -> LayerHeader:
        ni, no, th, r = struct.unpack("<IIII", data[:16])
        return cls(n_inputs=ni, n_outputs=no, threshold=th, reserved=r)

    @property
    def words_per_row(self) -> int:
        return (self.n_inputs + 31) // 32


def serialize_weights(layers: list[tuple[int, int, int, list[list[int]]]]) -> bytes:
    """Serialize network weights to binary blob.

    Parameters
    ----------
    layers : list
        Each entry is (n_inputs, n_outputs, threshold, weight_rows).
        weight_rows is list[list[int]] (n_outputs × words_per_row u32 values).

    Returns
    -------
    bytes
        Complete weight blob with headers.
    """
    header = WeightHeader(n_layers=len(layers))
    buf = bytearray(header.to_bytes())

    for n_inputs, n_outputs, threshold, rows in layers:
        lh = LayerHeader(n_inputs=n_inputs, n_outputs=n_outputs, threshold=threshold)
        buf.extend(lh.to_bytes())
        for row in rows:
            for word in row:
                buf.extend(struct.pack("<I", word & 0xFFFF_FFFF))

    return bytes(buf)


def deserialize_weights(data: bytes) -> list[tuple[LayerHeader, list[list[int]]]]:
    """Deserialize a weight blob into layer headers + weight matrices.

    Returns
    -------
    list[tuple[LayerHeader, list[list[int]]]]
        Each entry is (header, weight_rows).
    """
    header = WeightHeader.from_bytes(data[:16])
    if not header.validate():
        raise ValueError(f"Invalid weight blob: magic=0x{header.magic:08X}")

    offset = 16
    layers = []
    for _ in range(header.n_layers):
        lh = LayerHeader.from_bytes(data[offset : offset + 16])
        offset += 16
        rows = []
        wpr = lh.words_per_row
        for _ in range(lh.n_outputs):
            row = []
            for _ in range(wpr):
                (word,) = struct.unpack("<I", data[offset : offset + 4])
                row.append(word)
                offset += 4
            rows.append(row)
        layers.append((lh, rows))

    return layers

# SPDX-License-Identifier: AGPL-3.0-or-later
from typing import Any, Optional
import numpy as np
from dataclasses import dataclass


@dataclass
class DNAEncoder:
    """
    Interface for DNA Data Storage.
    Maps Bitstreams to Nucleotides (A, C, T, G).
    """

    mutation_rate: float = 0.001

    # Huffman-style mapping
    # 00 -> A, 01 -> C, 10 -> G, 11 -> T
    MAP = {(0, 0): "A", (0, 1): "C", (1, 0): "G", (1, 1): "T"}
    REV_MAP = {"A": (0, 0), "C": (0, 1), "G": (1, 0), "T": (1, 1)}

    def encode(self, bitstream: np.ndarray[Any, Any]) -> str:
        """
        Converts uint8 {0,1} bitstream to DNA string.
        """
        # Ensure even length
        if len(bitstream) % 2 != 0:
            bitstream = np.append(bitstream, 0)

        dna = []
        for i in range(0, len(bitstream), 2):
            pair = (bitstream[i], bitstream[i + 1])
            dna.append(self.MAP[pair])

        return "".join(dna)

    def decode(self, dna_str: str) -> np.ndarray[Any, Any]:
        """
        Converts DNA string back to bitstream.
        """
        bits: list[float] = []
        for char in dna_str:
            # Simulate mutation before decoding
            if np.random.random() < self.mutation_rate:
                char = np.random.choice(["A", "C", "T", "G"])

            pair = self.REV_MAP[char]
            bits.extend(pair)

        return np.array(bits, dtype=np.uint8)

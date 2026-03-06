# SPDX-License-Identifier: AGPL-3.0-or-later
from __future__ import annotations
from typing import Any
from dataclasses import dataclass
import numpy as np


@dataclass
class SCConv2DLayer:
    """
    Stochastic Computing 2D Convolutional Layer.

    Processes 2D input (e.g., images) using SC bitstreams.
    """

    in_channels: int
    out_channels: int
    kernel_size: int
    stride: int = 1
    padding: int = 0
    length: int = 256

    def __post_init__(self):  # type: ignore
        # Kernels: (out_channels, in_channels, k, k)
        self.kernels = np.random.uniform(
            0.0, 1.0, (self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)
        )

    def forward(self, input_image: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """
        input_image: (in_channels, H, W)
        Returns: (out_channels, H_out, W_out) as probabilities (or firing rates).
        """
        C_in, H, W = input_image.shape
        H_out = (H + 2 * self.padding - self.kernel_size) // self.stride + 1
        W_out = (W + 2 * self.padding - self.kernel_size) // self.stride + 1

        output = np.zeros((self.out_channels, H_out, W_out))

        # In a real SC hardware, this would be massive parallel AND-gates.
        # Here we simulate the probability math.

        for oc in range(self.out_channels):
            for ic in range(self.in_channels):
                # Apply padding
                padded_input = np.pad(input_image[ic], self.padding, mode="constant")

                for i in range(H_out):
                    for j in range(W_out):
                        h_start = i * self.stride
                        h_end = h_start + self.kernel_size
                        w_start = j * self.stride
                        w_end = w_start + self.kernel_size

                        region = padded_input[h_start:h_end, w_start:w_end]
                        kernel = self.kernels[oc, ic]

                        # SC Multiplication (AND) of probabilities
                        # For unipolar [0,1], P(A&B) = P(A)*P(B)
                        res = np.sum(region * kernel)
                        output[oc, i, j] += res

        # Normalize by kernel size and in_channels if needed,
        # or treat as accumulated current.
        return output

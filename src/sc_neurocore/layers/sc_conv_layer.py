# SPDX-License-Identifier: AGPL-3.0-or-later
from __future__ import annotations
from typing import Any
from dataclasses import dataclass
import numpy as np

from ..constants import LAYER_CONV_LENGTH


@dataclass
class SCConv2DLayer:
    """
    SC 2D convolutional layer using unipolar probability multiplication.

    Example
    -------
    >>> import numpy as np
    >>> conv = SCConv2DLayer(in_channels=1, out_channels=2, kernel_size=3, padding=1)
    >>> img = np.random.rand(1, 8, 8)
    >>> out = conv.forward(img)
    >>> out.shape
    (2, 8, 8)
    """

    in_channels: int
    out_channels: int
    kernel_size: int
    stride: int = 1
    padding: int = 0
    length: int = LAYER_CONV_LENGTH

    def __post_init__(self) -> None:
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
        if C_in != self.in_channels:
            raise IndexError(f"Expected {self.in_channels} input channels, got {C_in}")
        k = self.kernel_size
        H_out = (H + 2 * self.padding - k) // self.stride + 1
        W_out = (W + 2 * self.padding - k) // self.stride + 1

        if self.padding > 0:
            input_image = np.pad(
                input_image, ((0, 0), (self.padding, self.padding), (self.padding, self.padding))
            )

        # im2col: extract all patches → (H_out*W_out, C_in*k*k)
        col = np.empty((H_out * W_out, C_in * k * k), dtype=input_image.dtype)
        idx = 0
        for i in range(H_out):
            for j in range(W_out):
                hs = i * self.stride
                ws = j * self.stride
                col[idx] = input_image[:, hs : hs + k, ws : ws + k].ravel()
                idx += 1

        # SC multiply-accumulate: P(A&B) = P(A)*P(B) for unipolar [0,1]
        filters = self.kernels.reshape(self.out_channels, -1)  # (out, C_in*k*k)
        output = filters @ col.T  # (out, H_out*W_out)

        return output.reshape(self.out_channels, H_out, W_out)

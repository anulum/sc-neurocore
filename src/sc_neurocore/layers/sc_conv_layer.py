from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from ..accel._dispatch import njit_or_python


def _im2col(
    padded: np.ndarray, kernel_size: int, stride: int, H_out: int, W_out: int
) -> np.ndarray:
    """
    Extract sliding-window patches into a 2-D matrix (im2col).

    Args:
        padded: (C_in, H_padded, W_padded)

    Returns:
        cols: (C_in * kernel_size * kernel_size, H_out * W_out)
    """
    C_in = padded.shape[0]
    k = kernel_size
    cols = np.empty((C_in * k * k, H_out * W_out), dtype=padded.dtype)
    idx = 0
    for ic in range(C_in):
        for ki in range(k):
            for kj in range(k):
                cols[idx] = padded[
                    ic, ki : ki + stride * H_out : stride, kj : kj + stride * W_out : stride
                ].ravel()
                idx += 1
    return cols


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

    def __post_init__(self):
        # Kernels: (out_channels, in_channels, k, k)
        self.kernels = np.random.uniform(
            0.0, 1.0, (self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)
        )

    def forward(self, input_image: np.ndarray) -> np.ndarray:
        """
        input_image: (in_channels, H, W)
        Returns: (out_channels, H_out, W_out) as probabilities (or firing rates).
        """
        C_in, H, W = input_image.shape
        if C_in != self.in_channels:
            raise IndexError(f"Input has {C_in} channels but layer expects {self.in_channels}")
        H_out = (H + 2 * self.padding - self.kernel_size) // self.stride + 1
        W_out = (W + 2 * self.padding - self.kernel_size) // self.stride + 1

        # Pad all channels at once
        if self.padding > 0:
            padded = np.pad(
                input_image,
                ((0, 0), (self.padding, self.padding), (self.padding, self.padding)),
                mode="constant",
            )
        else:
            padded = input_image

        # im2col + matmul path (vectorized, no quadruple nested loop)
        cols = _im2col(padded, self.kernel_size, self.stride, H_out, W_out)
        kernels_flat = self.kernels.reshape(self.out_channels, -1)  # (OC, C_in*k*k)
        output = (kernels_flat @ cols).reshape(self.out_channels, H_out, W_out)

        return output

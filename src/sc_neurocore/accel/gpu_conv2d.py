"""
GPU-accelerated Conv2D using CuPy im2col + matmul.

Falls back to NumPy transparently when no GPU is available.
"""

from __future__ import annotations

import numpy as np
from .gpu_backend import xp, HAS_CUPY, to_device, to_host


def _im2col_xp(padded, kernel_size: int, stride: int, H_out: int, W_out: int):
    """Extract sliding-window patches using the active backend (xp)."""
    C_in = padded.shape[0]
    k = kernel_size
    cols = xp.empty((C_in * k * k, H_out * W_out), dtype=padded.dtype)
    idx = 0
    for ic in range(C_in):
        for ki in range(k):
            for kj in range(k):
                cols[idx] = padded[
                    ic, ki : ki + stride * H_out : stride, kj : kj + stride * W_out : stride
                ].ravel()
                idx += 1
    return cols


def gpu_conv2d_forward(
    input_image: np.ndarray,
    kernels: np.ndarray,
    stride: int = 1,
    padding: int = 0,
) -> np.ndarray:
    """
    Conv2D forward pass using im2col + matmul.

    Uses CuPy when available, otherwise pure NumPy.

    Args:
        input_image: (C_in, H, W)
        kernels: (out_channels, C_in, k, k)
        stride: convolution stride
        padding: zero-padding

    Returns:
        (out_channels, H_out, W_out)
    """
    C_in, H, W = input_image.shape
    out_channels = kernels.shape[0]
    k = kernels.shape[2]
    H_out = (H + 2 * padding - k) // stride + 1
    W_out = (W + 2 * padding - k) // stride + 1

    img = xp.asarray(input_image)
    kern = xp.asarray(kernels)

    if padding > 0:
        img = xp.pad(img, ((0, 0), (padding, padding), (padding, padding)), mode="constant")

    cols = _im2col_xp(img, k, stride, H_out, W_out)
    kern_flat = kern.reshape(out_channels, -1)
    result = kern_flat @ cols

    out = result.reshape(out_channels, H_out, W_out)
    return to_host(out) if HAS_CUPY else np.asarray(out)

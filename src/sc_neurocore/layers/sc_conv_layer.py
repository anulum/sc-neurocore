# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SC 2D convolutional layer using stochastic probability encodings

"""Stochastic-computing 2D convolution over channel-first image tensors.

The layer maps unipolar or bipolar probability encodings to deterministic
NumPy convolution accumulations. It validates construction parameters, input
rank, channel count, finite values, stochastic domains, and output geometry
before extracting im2col patches.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from ..constants import LAYER_CONV_LENGTH

SCMode = Literal["unipolar", "bipolar"]


@dataclass
class SCConv2DLayer:
    """
    SC 2D convolutional layer using stochastic probability encodings.

    ``sc_mode="unipolar"`` accepts probabilities in ``[0, 1]`` and uses
    probability-product accumulation. ``sc_mode="bipolar"`` accepts signed
    values in ``[-1, 1]`` and uses signed XNOR-equivalent products.

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
    sc_mode: SCMode = "unipolar"

    def __post_init__(self) -> None:
        """Validate the layer configuration and initialise stochastic kernels."""
        self._validate_configuration()
        low, high = (-1.0, 1.0) if self.sc_mode == "bipolar" else (0.0, 1.0)
        # Kernels: (out_channels, in_channels, k, k)
        self.kernels: NDArray[np.float64] = np.random.uniform(
            low,
            high,
            (self.out_channels, self.in_channels, self.kernel_size, self.kernel_size),
        )

    def _validate_configuration(self) -> None:
        if self.sc_mode not in ("unipolar", "bipolar"):
            raise ValueError("sc_mode must be 'unipolar' or 'bipolar'")
        if self.in_channels <= 0:
            raise ValueError("in_channels must be positive")
        if self.out_channels <= 0:
            raise ValueError("out_channels must be positive")
        if self.kernel_size <= 0:
            raise ValueError("kernel_size must be positive")
        if self.stride <= 0:
            raise ValueError("stride must be positive")
        if self.padding < 0:
            raise ValueError("padding must be non-negative")
        if self.length <= 0:
            raise ValueError("length must be positive")

    def _validate_input(self, input_image: NDArray[np.float64]) -> None:
        if input_image.ndim != 3:
            raise ValueError("input_image must have shape (in_channels, height, width)")
        if not np.all(np.isfinite(input_image)):
            raise ValueError("input_image must contain only finite values")
        c_in, height, width = input_image.shape
        if c_in != self.in_channels:
            raise IndexError(f"Expected {self.in_channels} input channels, got {c_in}")
        if height <= 0 or width <= 0:
            raise ValueError("input_image height and width must be positive")
        lower, upper = (-1.0, 1.0) if self.sc_mode == "bipolar" else (0.0, 1.0)
        if np.any((input_image < lower) | (input_image > upper)):
            raise ValueError(f"{self.sc_mode} input_image values must be in [{lower}, {upper}]")

    def _output_shape(self, height: int, width: int) -> tuple[int, int]:
        height_out = (height + 2 * self.padding - self.kernel_size) // self.stride + 1
        width_out = (width + 2 * self.padding - self.kernel_size) // self.stride + 1
        if height_out <= 0 or width_out <= 0:
            raise ValueError("kernel_size, stride, and padding produce an empty output")
        return height_out, width_out

    def forward(self, input_image: NDArray[Any]) -> NDArray[np.float64]:
        """Apply the stochastic convolution to a channel-first image tensor.

        Parameters
        ----------
        input_image:
            Input tensor shaped ``(in_channels, height, width)``. Values must
            be finite and belong to ``[0, 1]`` in unipolar mode or
            ``[-1, 1]`` in bipolar mode.

        Returns
        -------
        numpy.ndarray
            Accumulated convolution rates shaped
            ``(out_channels, height_out, width_out)``.

        Raises
        ------
        ValueError
            If the input rank, numeric domain, spatial dimensions, or output
            geometry is invalid.
        IndexError
            If the input channel count does not match ``in_channels``.
        """
        input_array = np.asarray(input_image, dtype=np.float64)
        self._validate_input(input_array)
        C_in, H, W = input_array.shape
        k = self.kernel_size
        H_out, W_out = self._output_shape(H, W)

        if self.padding > 0:
            input_array = np.pad(
                input_array, ((0, 0), (self.padding, self.padding), (self.padding, self.padding))
            )

        # im2col: extract all patches → (H_out*W_out, C_in*k*k)
        col = np.empty((H_out * W_out, C_in * k * k), dtype=np.float64)
        idx = 0
        for i in range(H_out):
            for j in range(W_out):
                hs = i * self.stride
                ws = j * self.stride
                col[idx] = input_array[:, hs : hs + k, ws : ws + k].ravel()
                idx += 1

        filters = self.kernels.reshape(self.out_channels, -1)  # (out, C_in*k*k)
        output = filters @ col.T  # (out, H_out*W_out)

        return output.reshape(self.out_channels, H_out, W_out)

"""
Phase 2a tests: im2col Conv2D equivalence.
"""

import numpy as np
import pytest

from sc_neurocore.layers.sc_conv_layer import SCConv2DLayer


class TestConv2DIm2col:
    """Verify im2col path matches naive 4-loop reference."""

    def _naive_conv2d(self, input_image, kernels, stride, padding):
        """Reference 4-loop implementation."""
        C_in, H, W = input_image.shape
        out_channels, _, ks, _ = kernels.shape
        H_out = (H + 2 * padding - ks) // stride + 1
        W_out = (W + 2 * padding - ks) // stride + 1
        output = np.zeros((out_channels, H_out, W_out))
        for oc in range(out_channels):
            for ic in range(C_in):
                padded = np.pad(input_image[ic], padding, mode="constant")
                for i in range(H_out):
                    for j in range(W_out):
                        h_s = i * stride
                        w_s = j * stride
                        region = padded[h_s : h_s + ks, w_s : w_s + ks]
                        output[oc, i, j] += np.sum(region * kernels[oc, ic])
        return output

    def test_im2col_matches_naive(self):
        np.random.seed(42)
        layer = SCConv2DLayer(in_channels=2, out_channels=3, kernel_size=3, stride=1, padding=1)
        img = np.random.random((2, 8, 8))

        result_im2col = layer.forward(img)
        result_naive = self._naive_conv2d(img, layer.kernels, layer.stride, layer.padding)

        np.testing.assert_allclose(result_im2col, result_naive, atol=1e-10)

    def test_no_padding(self):
        np.random.seed(42)
        layer = SCConv2DLayer(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=0)
        img = np.random.random((1, 6, 6))
        out = layer.forward(img)
        assert out.shape == (2, 4, 4)

    def test_stride_2(self):
        np.random.seed(42)
        layer = SCConv2DLayer(in_channels=1, out_channels=1, kernel_size=3, stride=2, padding=1)
        img = np.random.random((1, 8, 8))
        out = layer.forward(img)
        assert out.shape == (1, 4, 4)

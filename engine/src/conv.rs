// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — SC 2D convolutional layer using probability-domain

//! SC 2D convolutional layer using probability-domain multiplication.

use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;

/// Stochastic 2D convolutional layer.
///
/// Kernels stored as probabilities in [0,1]. Forward pass uses
/// P(A∧B) = P(A)·P(B) for unipolar stochastic multiplication.
#[derive(Clone, Debug)]
pub struct Conv2DLayer {
    pub in_channels: usize,
    pub out_channels: usize,
    pub kernel_size: usize,
    pub stride: usize,
    pub padding: usize,
    /// Flat kernels: [out_ch, in_ch, k, k]
    pub kernels: Vec<f64>,
}

impl Conv2DLayer {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        seed: u64,
    ) -> Self {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let size = out_channels * in_channels * kernel_size * kernel_size;
        let kernels: Vec<f64> = (0..size).map(|_| rng.random::<f64>()).collect();
        Self {
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            kernels,
        }
    }

    /// Forward pass: input shape [C_in, H, W] (flat), returns [C_out, H_out, W_out] (flat).
    pub fn forward(&self, input: &[f64], h: usize, w: usize) -> (Vec<f64>, usize, usize) {
        let k = self.kernel_size;
        let h_out = (h + 2 * self.padding - k) / self.stride + 1;
        let w_out = (w + 2 * self.padding - k) / self.stride + 1;
        let c_in = self.in_channels;
        let filter_size = c_in * k * k;

        // Pad input if needed
        let padded = if self.padding > 0 {
            let ph = h + 2 * self.padding;
            let pw = w + 2 * self.padding;
            let mut p = vec![0.0; c_in * ph * pw];
            for c in 0..c_in {
                for i in 0..h {
                    for j in 0..w {
                        p[c * ph * pw + (i + self.padding) * pw + (j + self.padding)] =
                            input[c * h * w + i * w + j];
                    }
                }
            }
            (p, ph, pw)
        } else {
            (input.to_vec(), h, w)
        };
        let (ref inp, ph, pw) = padded;

        let mut output = vec![0.0; self.out_channels * h_out * w_out];

        output
            .par_chunks_exact_mut(h_out * w_out)
            .enumerate()
            .for_each(|(oc, out_plane)| {
                let filter = &self.kernels[oc * filter_size..(oc + 1) * filter_size];
                for i in 0..h_out {
                    let mut j = 0;
                    while j + 3 < w_out {
                        let hs = i * self.stride;
                        let mut acc0 = 0.0;
                        let mut acc1 = 0.0;
                        let mut acc2 = 0.0;
                        let mut acc3 = 0.0;
                        for c in 0..c_in {
                            let input_offset = c * ph * pw;
                            let filter_offset = c * k * k;
                            for ki in 0..k {
                                let row_off = input_offset + (hs + ki) * pw;
                                let f_row_off = filter_offset + ki * k;
                                let filter_row = &filter[f_row_off..f_row_off + k];

                                acc0 += crate::simd::dot_f64_dispatch(
                                    &inp[row_off + j * self.stride..row_off + j * self.stride + k],
                                    filter_row,
                                );
                                acc1 += crate::simd::dot_f64_dispatch(
                                    &inp[row_off + (j + 1) * self.stride
                                        ..row_off + (j + 1) * self.stride + k],
                                    filter_row,
                                );
                                acc2 += crate::simd::dot_f64_dispatch(
                                    &inp[row_off + (j + 2) * self.stride
                                        ..row_off + (j + 2) * self.stride + k],
                                    filter_row,
                                );
                                acc3 += crate::simd::dot_f64_dispatch(
                                    &inp[row_off + (j + 3) * self.stride
                                        ..row_off + (j + 3) * self.stride + k],
                                    filter_row,
                                );
                            }
                        }
                        out_plane[i * w_out + j] = acc0;
                        out_plane[i * w_out + j + 1] = acc1;
                        out_plane[i * w_out + j + 2] = acc2;
                        out_plane[i * w_out + j + 3] = acc3;
                        j += 4;
                    }
                    while j < w_out {
                        let hs = i * self.stride;
                        let ws = j * self.stride;
                        let mut acc = 0.0;
                        for c in 0..c_in {
                            let input_offset = c * ph * pw;
                            let filter_offset = c * k * k;
                            for ki in 0..k {
                                let inp_row = &inp[input_offset + (hs + ki) * pw + ws
                                    ..input_offset + (hs + ki) * pw + ws + k];
                                let filter_row =
                                    &filter[filter_offset + ki * k..filter_offset + (ki + 1) * k];
                                acc += crate::simd::dot_f64_dispatch(inp_row, filter_row);
                            }
                        }
                        out_plane[i * w_out + j] = acc;
                        j += 1;
                    }
                }
            });

        (output, h_out, w_out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn output_shape_no_padding() {
        let conv = Conv2DLayer::new(1, 2, 3, 1, 0, 42);
        let input = vec![0.5; 8 * 8];
        let (out, h, w) = conv.forward(&input, 8, 8);
        assert_eq!(h, 6);
        assert_eq!(w, 6);
        assert_eq!(out.len(), 2 * 6 * 6);
    }

    #[test]
    fn output_shape_with_padding() {
        let conv = Conv2DLayer::new(1, 2, 3, 1, 1, 42);
        let input = vec![0.5; 8 * 8];
        let (out, h, w) = conv.forward(&input, 8, 8);
        assert_eq!(h, 8);
        assert_eq!(w, 8);
        assert_eq!(out.len(), 2 * 8 * 8);
    }

    #[test]
    fn all_ones_kernel() {
        let mut conv = Conv2DLayer::new(1, 1, 3, 1, 0, 42);
        conv.kernels = vec![1.0; 9];
        let input = vec![1.0; 5 * 5];
        let (out, _, _) = conv.forward(&input, 5, 5);
        // Each output = sum of 3×3 patch of 1.0 × 1.0 = 9.0
        assert!((out[0] - 9.0).abs() < 1e-10);
    }

    #[test]
    fn stride_2() {
        let conv = Conv2DLayer::new(1, 1, 3, 2, 0, 42);
        let input = vec![0.5; 8 * 8];
        let (_, h, w) = conv.forward(&input, 8, 8);
        assert_eq!(h, 3);
        assert_eq!(w, 3);
    }
}

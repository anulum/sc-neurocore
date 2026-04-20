# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for layers/sc_conv_layer

module ScConvLayerAccel

using Statistics, LinearAlgebra

mutable struct SCConv2DLayerState
    in_channels::Float64
    out_channels::Float64
    kernel_size::Float64
    stride::Float64
    padding::Float64
    length::Float64
end

function SCConv2DLayerState()
    SCConv2DLayerState(0.0, 0.0, 0.0, 1.0, 0.0, 0.0)
end

function forward(s::SCConv2DLayerState, input_image, Any])
    C_in, H, W = input_image.shape
    if C_in != s.in_channels
        raise IndexError(f"Expected {s.in_channels} input channels, got {C_in}")
    k = s.kernel_size
    H_out = (H + 2 * s.padding - k) // s.stride + 1
    W_out = (W + 2 * s.padding - k) // s.stride + 1
    if s.padding > 0
        input_image = np.pad(
            input_image, ((0, 0), (s.padding, s.padding), (s.padding, s.padding))
        )
    # im2col: extract all patches → (H_out*W_out, C_in*k*k)
    col = np.empty((H_out * W_out, C_in * k * k), dtype=input_image.dtype)
    idx = 0
    for i in 1:H_out
        for j in 1:W_out
            hs = i * s.stride
            ws = j * s.stride
            col[idx] = input_image[:, hs : hs + k, ws : ws + k].ravel()
            idx += 1
    # SC multiply-accumulate: P(A&B) = P(A)*P(B) for unipolar [0,1]
    filters = s.kernels.reshape(s.out_channels, -1)  # (out, C_in*k*k)
    output = filters @ col.T  # (out, H_out*W_out)
    return output.reshape(s.out_channels, H_out, W_out)
end

end # module ScConvLayerAccel

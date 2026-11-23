# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo C-ABI lane for contrastive/ssl (InfoNCE loss, CSDP update, goodness)

from std.math import exp, isfinite, log, sqrt
from std.memory import UnsafePointer

# Status codes shared with the Python facade:
# 0 success; 1 invalid arguments (negative count, missing buffer or
# overlapping output); 2 a non-finite input value.


@always_inline
def _ranges_overlap(a_addr: Int, a_elements: Int, b_addr: Int, b_elements: Int) -> Bool:
    if a_elements == 0 or b_elements == 0:
        return False
    var a_bytes = a_elements * 8
    var b_bytes = b_elements * 8
    if a_addr <= b_addr:
        return b_addr - a_addr < a_bytes
    return a_addr - b_addr < b_bytes


def _all_finite(values: UnsafePointer[Float64, MutAnyOrigin], count: Int) -> Bool:
    for index in range(count):
        if not isfinite(values[index]):
            return False
    return True


@export
def contrastive_ssl_infonce_c(
    batch: Int32,
    features: Int32,
    temperature: Float64,
    view_a_addr: Int,
    view_b_addr: Int,
    loss_addr: Int,
) -> Int32:
    """Write the mean InfoNCE loss of two row-major (batch, features) views to ``loss_addr``."""
    if batch < 0 or features < 1 or view_a_addr == 0 or view_b_addr == 0 or loss_addr == 0:
        return 1
    var rows = Int(batch)
    var cols = Int(features)
    if _ranges_overlap(loss_addr, 1, view_a_addr, rows * cols) or _ranges_overlap(
        loss_addr, 1, view_b_addr, rows * cols
    ):
        return 1
    if not isfinite(temperature) or temperature <= 0.0:
        return 2
    var view_a = UnsafePointer[Float64, MutAnyOrigin](unsafe_from_address=view_a_addr)
    var view_b = UnsafePointer[Float64, MutAnyOrigin](unsafe_from_address=view_b_addr)
    var loss = UnsafePointer[Float64, MutAnyOrigin](unsafe_from_address=loss_addr)
    if not _all_finite(view_a, rows * cols) or not _all_finite(view_b, rows * cols):
        return 2
    if rows < 2:
        loss[0] = 0.0
        return 0
    # Normalise each row first, as the Python original does, then take dot products.
    var a_unit = List[Float64](capacity=rows * cols)
    var b_unit = List[Float64](capacity=rows * cols)
    for row in range(rows):
        var a_sum = 0.0
        var b_sum = 0.0
        for col in range(cols):
            var a_value = view_a[row * cols + col]
            var b_value = view_b[row * cols + col]
            a_sum += a_value * a_value
            b_sum += b_value * b_value
        var a_length = max(sqrt(a_sum), 1.0e-8)
        var b_length = max(sqrt(b_sum), 1.0e-8)
        for col in range(cols):
            a_unit.append(view_a[row * cols + col] / a_length)
            b_unit.append(view_b[row * cols + col] / b_length)
    var total = 0.0
    var logits = List[Float64](capacity=rows)
    for row in range(rows):
        logits.clear()
        var largest = -1.0e308
        for other in range(rows):
            var dot = 0.0
            for col in range(cols):
                dot += a_unit[row * cols + col] * b_unit[other * cols + col]
            var logit = dot / temperature
            logits.append(logit)
            if logit > largest:
                largest = logit
        var denominator = 0.0
        for other in range(rows):
            denominator += exp(logits[other] - largest)
        var probability = exp(logits[row] - largest) / denominator
        total += log(max(probability, 1.0e-10))
    loss[0] = -total / Float64(rows)
    return 0


@export
def contrastive_ssl_csdp_c(
    n_post: Int32,
    n_pre: Int32,
    lr: Float64,
    decay: Float64,
    positive: Int32,
    weights_addr: Int,
    pre_addr: Int,
    post_addr: Int,
    out_addr: Int,
) -> Int32:
    """Write one CSDP phase of the row-major (n_post, n_pre) weights to ``out_addr``.

    ``positive`` selects the Hebbian phase ``W + lr*outer(post, pre) - decay*W``;
    otherwise the anti-Hebbian phase ``W - lr*outer(post, pre)`` is written.
    """
    if n_post < 0 or n_pre < 0 or positive < 0 or positive > 1:
        return 1
    var rows = Int(n_post)
    var cols = Int(n_pre)
    var cells = rows * cols
    if cells > 0 and (weights_addr == 0 or out_addr == 0):
        return 1
    if cols > 0 and pre_addr == 0:
        return 1
    if rows > 0 and post_addr == 0:
        return 1
    if _ranges_overlap(out_addr, cells, pre_addr, cols) or _ranges_overlap(out_addr, cells, post_addr, rows):
        return 1
    if not isfinite(lr) or lr < 0.0 or not isfinite(decay) or decay < 0.0:
        return 2
    var weights = UnsafePointer[Float64, MutAnyOrigin](unsafe_from_address=weights_addr)
    var pre = UnsafePointer[Float64, MutAnyOrigin](unsafe_from_address=pre_addr)
    var post = UnsafePointer[Float64, MutAnyOrigin](unsafe_from_address=post_addr)
    var out = UnsafePointer[Float64, MutAnyOrigin](unsafe_from_address=out_addr)
    if not _all_finite(weights, cells) or not _all_finite(pre, cols) or not _all_finite(post, rows):
        return 2
    for row in range(rows):
        for col in range(cols):
            var index = row * cols + col
            var hebbian = lr * (post[row] * pre[col])
            if positive == 1:
                out[index] = weights[index] + (hebbian - decay * weights[index])
            else:
                out[index] = weights[index] + (-hebbian)
    return 0


@export
def contrastive_ssl_goodness_c(count: Int64, values_addr: Int, goodness_addr: Int) -> Int32:
    """Write the sum of squared activations to ``goodness_addr``."""
    if count < 0 or goodness_addr == 0 or (count > 0 and values_addr == 0):
        return 1
    var n = Int(count)
    if _ranges_overlap(goodness_addr, 1, values_addr, n):
        return 1
    var values = UnsafePointer[Float64, MutAnyOrigin](unsafe_from_address=values_addr)
    var goodness = UnsafePointer[Float64, MutAnyOrigin](unsafe_from_address=goodness_addr)
    if not _all_finite(values, n):
        return 2
    var total = 0.0
    for index in range(n):
        total += values[index] * values[index]
    goodness[0] = total
    return 0

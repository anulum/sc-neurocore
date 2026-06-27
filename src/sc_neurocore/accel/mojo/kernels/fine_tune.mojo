# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo transfer fine-tune validation kernel

from std.math import isfinite


def contains(values: List[Int], target: Int) -> Bool:
    for idx in range(len(values)):
        if values[idx] == target:
            return True
    return False


def freeze_until(layer_count: Int, frozen: List[Int], until_index: Int) raises -> List[Int]:
    if until_index < 0 or until_index >= layer_count:
        raise Error("until_index must reference an existing layer")
    var out = List[Int]()
    for idx in range(len(frozen)):
        if frozen[idx] < 0 or frozen[idx] >= layer_count:
            raise Error("frozen layer must reference an existing layer")
        if not contains(out, frozen[idx]):
            out.append(frozen[idx])
    for idx in range(until_index + 1):
        if not contains(out, idx):
            out.append(idx)
    return out^


def apply_transfer_config(
    layer_count: Int,
    freeze_index: Int,
    lr_backbone: Float64,
    lr_head: Float64,
) raises -> List[Float64]:
    if not isfinite(lr_backbone) or lr_backbone < 0.0:
        raise Error("lr_backbone must be finite and non-negative")
    if not isfinite(lr_head) or lr_head < 0.0:
        raise Error("lr_head must be finite and non-negative")
    var frozen = List[Int]()
    if freeze_index >= 0:
        frozen = freeze_until(layer_count, frozen, freeze_index)
    var rates = List[Float64]()
    for layer in range(layer_count):
        if contains(frozen, layer):
            rates.append(lr_backbone)
        else:
            rates.append(lr_head)
    return rates^


def validate_fine_tune_kernel() raises -> Bool:
    var rates = apply_transfer_config(2, 0, 0.0, 0.01)
    if len(rates) != 2:
        return False
    if rates[0] != 0.0 or rates[1] != 0.01:
        return False
    try:
        _ = apply_transfer_config(2, 4, 0.0, 0.01)
    except:
        return True
    return False


def main() raises:
    if not validate_fine_tune_kernel():
        raise Error("transfer fine-tune validation failed")

# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo transfer checkpoint validation kernel

from std.math import isfinite


def vector1(value: Float64) -> List[Float64]:
    var row = List[Float64]()
    row.append(value)
    return row^


def vector2(a: Float64, b: Float64) -> List[Float64]:
    var row = List[Float64]()
    row.append(a)
    row.append(b)
    return row^


def validate_weight_layer(
    weight: List[List[Float64]], inputs: Int, outputs: Int
) -> Bool:
    if len(weight) != outputs:
        return False
    for row in range(len(weight)):
        if len(weight[row]) != inputs:
            return False
        for col in range(len(weight[row])):
            if not isfinite(weight[row][col]):
                return False
    return True


def total_params(weights: List[List[List[Float64]]]) -> Int:
    var total = 0
    for layer in range(len(weights)):
        for row in range(len(weights[layer])):
            total += len(weights[layer][row])
    return total


def validate_checkpoint(weights: List[List[List[Float64]]]) -> Bool:
    if len(weights) != 2:
        return False
    if not validate_weight_layer(weights[0], 2, 2):
        return False
    if not validate_weight_layer(weights[1], 2, 1):
        return False
    return total_params(weights) == 6


def checkpoint_fixture() -> List[List[List[Float64]]]:
    var weights = List[List[List[Float64]]]()
    var hidden = List[List[Float64]]()
    hidden.append(vector2(0.1, 0.2))
    hidden.append(vector2(0.3, 0.4))
    var output = List[List[Float64]]()
    output.append(vector2(0.5, 0.6))
    weights.append(hidden^)
    weights.append(output^)
    return weights^


def validate_checkpoint_kernel() -> Bool:
    var weights = checkpoint_fixture()
    if not validate_checkpoint(weights):
        return False
    var bad = List[List[List[Float64]]]()
    var bad_layer = List[List[Float64]]()
    bad_layer.append(vector1(0.1))
    bad.append(bad_layer^)
    bad.append(weights[1].copy())
    return not validate_checkpoint(bad)


def main() raises:
    if not validate_checkpoint_kernel():
        raise Error("transfer checkpoint validation failed")

# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo owned dense IF parameters

"""Owned finite dense layers and connected inference stacks."""

from std.math import isfinite
from ann_to_snn_resources import admit_copy, checked_add


def validate_dense(inputs: Int, outputs: Int, weights: List[Float64], bias: List[Float64], threshold: Float64, initial_fraction: Float64, maximum: Int) raises:
    """Validate dense geometry, coefficient reservation and finite domains.

    Args:
        inputs: Positive input width.
        outputs: Positive output width.
        weights: Flat row-major coefficients.
        bias: Output currents or empty absent bias.
        threshold: Positive inclusive threshold.
        initial_fraction: Finite membrane preload fraction.
        maximum: Positive numeric copy reservation limit.

    Raises:
        Error: Invalid geometry/domain or exceeded numeric reservation.
    """
    if inputs <= 0 or outputs <= 0 or inputs > 0x7FFFFFFFFFFFFFFF // outputs:
        raise Error("IF invalid input")
    if inputs * outputs != len(weights) or (len(bias) != 0 and len(bias) != outputs):
        raise Error("IF invalid input")
    admit_copy(checked_add(len(weights), len(bias)), maximum)
    if not isfinite(threshold) or threshold <= 0 or not isfinite(initial_fraction):
        raise Error("IF invalid input")
    for value in weights:
        if not isfinite(value):
            raise Error("IF invalid input")
    for value in bias:
        if not isfinite(value):
            raise Error("IF invalid input")


struct DenseLayer(Copyable, Movable):
    """Row-major output-by-input coefficients with a finite per-layer preload."""
    var inputs: Int
    """Positive input width."""
    var outputs: Int
    """Positive output width."""
    var weights: List[Float64]
    """Owned flat row-major output-by-input coefficients."""
    var bias: List[Float64]
    """Owned per-step output currents; empty means absent bias."""
    var threshold: Float64
    """Finite positive inclusive IF threshold."""
    var initial_fraction: Float64
    """Finite default membrane preload in threshold units."""

    def __init__(out self, inputs: Int, outputs: Int, weights: List[Float64], bias: List[Float64], threshold: Float64, initial_fraction: Float64 = 0.0, max_working_bytes: Int = 268435456) raises:
        """Validate geometry, numeric copy budget and finite domains before ownership.

        Args:
            inputs: Positive dense input width.
            outputs: Positive dense output width.
            weights: Flat row-major output-by-input finite doubles.
            bias: Finite per-step output currents or empty for absent bias.
            threshold: Positive finite inclusive firing threshold.
            initial_fraction: Finite membrane preload in threshold units.
            max_working_bytes: Positive numeric copy budget excluding caller storage.

        Raises:
            Error: Invalid geometry/domain or exceeded numeric copy reservation.
        """
        validate_dense(inputs, outputs, weights, bias, threshold, initial_fraction, max_working_bytes)
        self.inputs = inputs
        self.outputs = outputs
        self.weights = weights.copy()
        self.bias = bias.copy()
        self.threshold = threshold
        self.initial_fraction = initial_fraction


    def __init__(out self, inputs: Int, outputs: Int, *, var owned_weights: List[Float64], var owned_bias: List[Float64], threshold: Float64, initial_fraction: Float64 = 0.0, max_working_bytes: Int = 268435456) raises:
        """Validate and consume already admitted coefficient owners without copying.

        Args:
            inputs: Positive input width.
            outputs: Positive output width.
            owned_weights: Independently owned row-major coefficient storage to transfer.
            owned_bias: Independently owned current storage to transfer.
            threshold: Positive finite inclusive threshold.
            initial_fraction: Finite membrane preload fraction.
            max_working_bytes: Positive numeric coefficient reservation limit.

        Raises:
            Error: Invalid geometry/domain or exceeded numeric reservation.
        """
        validate_dense(inputs, outputs, owned_weights, owned_bias, threshold, initial_fraction, max_working_bytes)
        self.inputs = inputs
        self.outputs = outputs
        self.weights = owned_weights^
        self.bias = owned_bias^
        self.threshold = threshold
        self.initial_fraction = initial_fraction


struct ConvertedSNN(Copyable, Movable):
    """Owned connected stack; linear final mode retains signed cumulative current."""
    var layers: List[DenseLayer]
    """Owned connected dense stages."""
    var linear: Bool
    """True selects final signed cumulative current instead of IF events."""

    def __init__(out self, layers: List[DenseLayer], linear: Bool = False, max_working_bytes: Int = 268435456) raises:
        """Admit and independently own a nonempty connected inference stack.

        Args:
            layers: Nonempty connected dense stages with finite parameters.
            linear: Select final signed linear integration when true.
            max_working_bytes: Positive numeric coefficient-copy limit.

        Raises:
            Error: Invalid stack, mutated parameters or numeric copy refusal.
        """
        if len(layers) == 0:
            raise Error("IF invalid input")
        var coefficients = 0
        for i in range(len(layers)):
            if i > 0 and layers[i-1].outputs != layers[i].inputs:
                raise Error("IF invalid input")
            coefficients = checked_add(coefficients, checked_add(len(layers[i].weights), len(layers[i].bias)))
        admit_copy(coefficients, max_working_bytes)
        self.layers = List[DenseLayer]()
        for i in range(len(layers)):
            self.layers.append(DenseLayer(layers[i].inputs, layers[i].outputs, layers[i].weights, layers[i].bias, layers[i].threshold, layers[i].initial_fraction, max_working_bytes))
        self.linear = linear


    def __init__(out self, *, var owned_layers: List[DenseLayer], linear: Bool = False, max_working_bytes: Int = 268435456) raises:
        """Validate and consume an independently owned connected stack without copying.

        Args:
            owned_layers: Nonempty independently owned connected dense stages to transfer.
            linear: Select final signed linear integration.
            max_working_bytes: Positive numeric coefficient reservation limit.

        Raises:
            Error: Invalid stack, coefficient domains or reservation refusal.
        """
        if len(owned_layers) == 0:
            raise Error("IF invalid input")
        var coefficients = 0
        for i in range(len(owned_layers)):
            if i > 0 and owned_layers[i-1].outputs != owned_layers[i].inputs:
                raise Error("IF invalid input")
            coefficients = checked_add(coefficients, checked_add(len(owned_layers[i].weights), len(owned_layers[i].bias)))
        admit_copy(coefficients, max_working_bytes)
        for layer in owned_layers:
            validate_dense(layer.inputs, layer.outputs, layer.weights, layer.bias, layer.threshold, layer.initial_fraction, max_working_bytes)
        self.layers = owned_layers^
        self.linear = linear

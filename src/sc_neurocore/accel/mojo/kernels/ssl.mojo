# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo contrastive SSL validation kernels

from std.math import exp, isfinite, log, sqrt


def validate_vector(values: List[Float64]) -> Bool:
    for idx in range(len(values)):
        if not isfinite(values[idx]):
            return False
    return True


def validate_matrix(values: List[List[Float64]]) -> Bool:
    if len(values) == 0:
        return True
    var n_features = len(values[0])
    if n_features == 0:
        return False
    for row in range(len(values)):
        if len(values[row]) != n_features:
            return False
        if not validate_vector(values[row]):
            return False
    return True


def dot(lhs: List[Float64], rhs: List[Float64]) -> Float64:
    var total = 0.0
    for idx in range(len(lhs)):
        total += lhs[idx] * rhs[idx]
    return total


def row_norm(values: List[Float64]) -> Float64:
    var squared = 0.0
    for idx in range(len(values)):
        squared += values[idx] * values[idx]
    var denom = sqrt(squared)
    if denom <= 1.0e-8:
        return 1.0e-8
    return denom


def normalise_rows(values: List[List[Float64]]) -> List[List[Float64]]:
    var out = List[List[Float64]]()
    for row in range(len(values)):
        var denom = row_norm(values[row])
        var normalised = List[Float64]()
        for col in range(len(values[row])):
            normalised.append(values[row][col] / denom)
        out.append(normalised^)
    return out^


struct SpikeContrastiveLoss:
    var temperature: Float64

    def __init__(out self, temperature: Float64) raises:
        if not isfinite(temperature) or temperature <= 0.0:
            raise Error("temperature must be finite and positive")
        self.temperature = temperature

    def compute(
        self, view_a: List[List[Float64]], view_b: List[List[Float64]]
    ) raises -> Float64:
        if len(view_a) != len(view_b):
            raise Error("view_a and view_b must have the same shape")
        if not validate_matrix(view_a) or not validate_matrix(view_b):
            raise Error(
                "views must be finite matrices with at least one feature"
            )
        if len(view_a) > 0 and len(view_a[0]) != len(view_b[0]):
            raise Error("view_a and view_b must have the same shape")
        if len(view_a) < 2:
            return 0.0

        var a_norm = normalise_rows(view_a)
        var b_norm = normalise_rows(view_b)
        var total = 0.0
        for row in range(len(a_norm)):
            var logits = List[Float64]()
            var row_max = -1.0e300
            for rhs in range(len(b_norm)):
                var logit = dot(a_norm[row], b_norm[rhs]) / self.temperature
                logits.append(logit)
                if logit > row_max:
                    row_max = logit

            var denom = 0.0
            var positive = 0.0
            for idx in range(len(logits)):
                var value = exp(logits[idx] - row_max)
                if idx == row:
                    positive = value
                denom += value
            var probability = positive / denom
            if probability < 1.0e-10:
                probability = 1.0e-10
            total += log(probability)
        return -total / Float64(len(a_norm))


struct CSDPRule:
    var lr: Float64
    var decay: Float64

    def __init__(out self, lr: Float64, decay: Float64) raises:
        if not isfinite(lr) or lr < 0.0:
            raise Error("lr must be finite and non-negative")
        if not isfinite(decay) or decay < 0.0:
            raise Error("decay must be finite and non-negative")
        self.lr = lr
        self.decay = decay

    def validate_update(
        self,
        weights: List[List[Float64]],
        pre_spikes: List[Float64],
        post_spikes: List[Float64],
    ) raises:
        if not validate_matrix(weights):
            raise Error("weights must be a finite matrix")
        if not validate_vector(pre_spikes) or not validate_vector(post_spikes):
            raise Error("spike vectors must contain only finite values")
        if len(weights) != len(post_spikes):
            raise Error("weights must have len(post_spikes) rows")
        for row in range(len(weights)):
            if len(weights[row]) != len(pre_spikes):
                raise Error("weights rows must have len(pre_spikes) columns")

    def positive_update(
        self,
        weights: List[List[Float64]],
        pre_spikes: List[Float64],
        post_spikes: List[Float64],
    ) raises -> List[List[Float64]]:
        self.validate_update(weights, pre_spikes, post_spikes)
        var out = List[List[Float64]]()
        for post_idx in range(len(post_spikes)):
            var row = List[Float64]()
            for pre_idx in range(len(pre_spikes)):
                row.append(
                    weights[post_idx][pre_idx]
                    + self.lr * post_spikes[post_idx] * pre_spikes[pre_idx]
                    - self.decay * weights[post_idx][pre_idx]
                )
            out.append(row^)
        return out^

    def negative_update(
        self,
        weights: List[List[Float64]],
        pre_spikes: List[Float64],
        post_spikes: List[Float64],
    ) raises -> List[List[Float64]]:
        self.validate_update(weights, pre_spikes, post_spikes)
        var out = List[List[Float64]]()
        for post_idx in range(len(post_spikes)):
            var row = List[Float64]()
            for pre_idx in range(len(pre_spikes)):
                row.append(
                    weights[post_idx][pre_idx]
                    - self.lr * post_spikes[post_idx] * pre_spikes[pre_idx]
                )
            out.append(row^)
        return out^

    def contrastive_step(
        self,
        weights: List[List[Float64]],
        pos_pre: List[Float64],
        pos_post: List[Float64],
        neg_pre: List[Float64],
        neg_post: List[Float64],
    ) raises -> List[List[Float64]]:
        var after_positive = self.positive_update(weights, pos_pre, pos_post)
        return self.negative_update(after_positive, neg_pre, neg_post)

    def goodness(self, activations: List[Float64]) raises -> Float64:
        if not validate_vector(activations):
            raise Error("activations must contain only finite values")
        var total = 0.0
        for idx in range(len(activations)):
            total += activations[idx] * activations[idx]
        return total


def vector2(a: Float64, b: Float64) -> List[Float64]:
    var values = List[Float64]()
    values.append(a)
    values.append(b)
    return values^


def vector3(a: Float64, b: Float64, c: Float64) -> List[Float64]:
    var values = List[Float64]()
    values.append(a)
    values.append(b)
    values.append(c)
    return values^


def validate_ssl() raises -> Bool:
    var loss = SpikeContrastiveLoss(0.5)
    var view = List[List[Float64]]()
    view.append(vector3(1.0, 0.0, 0.0))
    view.append(vector3(0.0, 1.0, 0.0))
    view.append(vector3(0.0, 0.0, 1.0))
    if loss.compute(view, view) < 0.0:
        return False

    var rule = CSDPRule(0.1, 0.01)
    var weights = List[List[Float64]]()
    weights.append(vector2(0.2, 0.4))
    weights.append(vector2(0.1, 0.3))
    var updated = rule.contrastive_step(
        weights,
        vector2(1.0, 0.5),
        vector2(0.25, 1.0),
        vector2(0.0, 1.0),
        vector2(0.5, 0.5),
    )
    if len(updated) != len(weights) or len(updated[0]) != len(weights[0]):
        return False
    return abs(rule.goodness(vector3(1.0, -2.0, 0.5)) - 5.25) < 1.0e-12


def main() raises:
    if not validate_ssl():
        raise Error("contrastive SSL validation failed")

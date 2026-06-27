# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo few-shot HAAM validation kernels

from std.math import isfinite, sqrt


def cosine_score(lhs: List[Float64], rhs: List[Float64]) -> Float64:
    var dot = 0.0
    var lhs_sq = 0.0
    var rhs_sq = 0.0
    for idx in range(len(lhs)):
        dot += lhs[idx] * rhs[idx]
        lhs_sq += lhs[idx] * lhs[idx]
        rhs_sq += rhs[idx] * rhs[idx]
    var denom = sqrt(lhs_sq) * sqrt(rhs_sq)
    if denom <= 1.0e-12:
        return 0.0
    return dot / denom


def euclidean_score(lhs: List[Float64], rhs: List[Float64]) -> Float64:
    var squared = 0.0
    for idx in range(len(lhs)):
        var diff = lhs[idx] - rhs[idx]
        squared += diff * diff
    return -sqrt(squared)


def hamming_score(lhs: List[Float64], rhs: List[Float64]) -> Float64:
    var disagreements = 0
    for idx in range(len(lhs)):
        if (lhs[idx] > 0.0) != (rhs[idx] > 0.0):
            disagreements += 1
    return -Float64(disagreements) / Float64(len(lhs))


def validate_pattern(pattern: List[Float64], n_features: Int) -> Bool:
    if len(pattern) != n_features:
        return False
    for idx in range(len(pattern)):
        if not isfinite(pattern[idx]):
            return False
    return True


def temporal_mean(
    train: List[List[Float64]], n_features: Int
) raises -> List[Float64]:
    if len(train) == 0:
        raise Error("temporal spike train must contain at least one timestep")
    var out = List[Float64]()
    for _ in range(n_features):
        out.append(0.0)
    for step in range(len(train)):
        if not validate_pattern(train[step], n_features):
            raise Error("temporal spike train has invalid feature shape")
        for feature in range(n_features):
            out[feature] += train[step][feature]
    for feature in range(n_features):
        out[feature] = out[feature] / Float64(len(train))
    return out^


struct HebbianFewShot:
    var n_features: Int
    var n_classes: Int
    var lr_hebbian: Float64
    var memory: List[Float64]
    var counts: List[Int]

    def __init__(
        out self, n_features: Int, n_classes: Int, lr_hebbian: Float64
    ) raises:
        if n_features <= 0:
            raise Error("n_features must be positive")
        if n_classes <= 0:
            raise Error("n_classes must be positive")
        if not isfinite(lr_hebbian) or lr_hebbian < 0.0:
            raise Error("lr_hebbian must be finite and non-negative")
        self.n_features = n_features
        self.n_classes = n_classes
        self.lr_hebbian = lr_hebbian
        self.memory = List[Float64]()
        self.counts = List[Int]()
        for _ in range(n_classes * n_features):
            self.memory.append(0.0)
        for _ in range(n_classes):
            self.counts.append(0)

    def store(mut self, pattern: List[Float64], label: Int) raises:
        if label < 0 or label >= self.n_classes:
            raise Error("label out of range")
        if not validate_pattern(pattern, self.n_features):
            raise Error(
                "spike_pattern must match n_features and contain finite values"
            )
        var row_start = label * self.n_features
        for feature in range(self.n_features):
            self.memory[row_start + feature] += (
                self.lr_hebbian * pattern[feature]
            )
        self.counts[label] += 1

    def store_temporal(mut self, train: List[List[Float64]], label: Int) raises:
        self.store(temporal_mean(train, self.n_features), label)

    def query_scores(self, pattern: List[Float64]) raises -> List[Float64]:
        if not validate_pattern(pattern, self.n_features):
            raise Error(
                "spike_pattern must match n_features and contain finite values"
            )
        var scores = List[Float64]()
        for _ in range(self.n_classes):
            scores.append(0.0)
        for class_idx in range(self.n_classes):
            if self.counts[class_idx] == 0:
                continue
            var memory_row = List[Float64]()
            var row_start = class_idx * self.n_features
            for feature in range(self.n_features):
                memory_row.append(self.memory[row_start + feature])
            scores[class_idx] = cosine_score(memory_row, pattern)
        return scores^

    def query(self, pattern: List[Float64]) raises -> Int:
        var has_support = False
        for class_idx in range(self.n_classes):
            if self.counts[class_idx] > 0:
                has_support = True
        if not has_support:
            raise Error(
                "at least one support example must be stored before query"
            )
        var scores = self.query_scores(pattern)
        var best_class = 0
        var best_score = scores[0]
        for class_idx in range(1, self.n_classes):
            if scores[class_idx] > best_score:
                best_score = scores[class_idx]
                best_class = class_idx
        return best_class

    def reset(mut self):
        for idx in range(len(self.memory)):
            self.memory[idx] = 0.0
        for idx in range(len(self.counts)):
            self.counts[idx] = 0


struct SpikePrototypeNet:
    var n_features: Int
    var metric: Int
    var prototypes: List[List[Float64]]
    var labels: List[Int]

    def __init__(out self, n_features: Int, metric: Int) raises:
        if n_features <= 0:
            raise Error("n_features must be positive")
        if metric < 0 or metric > 2:
            raise Error("metric must be 0 cosine, 1 euclidean, or 2 hamming")
        self.n_features = n_features
        self.metric = metric
        self.prototypes = List[List[Float64]]()
        self.labels = List[Int]()

    def classify(
        mut self,
        support_x: List[List[Float64]],
        support_y: List[Int],
        query_x: List[List[Float64]],
    ) raises -> List[Int]:
        self.build_prototypes(support_x, support_y)
        var predictions = List[Int]()
        for query_idx in range(len(query_x)):
            if not validate_pattern(query_x[query_idx], self.n_features):
                raise Error(
                    "query must match n_features and contain finite values"
                )
            var best_idx = 0
            var best_score = self.metric_score(
                query_x[query_idx], self.prototypes[0]
            )
            for proto_idx in range(1, len(self.prototypes)):
                var score = self.metric_score(
                    query_x[query_idx], self.prototypes[proto_idx]
                )
                if score > best_score:
                    best_score = score
                    best_idx = proto_idx
            predictions.append(self.labels[best_idx])
        return predictions^

    def build_prototypes(
        mut self, support_x: List[List[Float64]], support_y: List[Int]
    ) raises:
        if len(support_x) == 0:
            raise Error("support_x must contain at least one support pattern")
        if len(support_x) != len(support_y):
            raise Error("support_x and support_y must have the same length")
        self.prototypes = List[List[Float64]]()
        self.labels = List[Int]()
        for support_idx in range(len(support_x)):
            if support_y[support_idx] < 0:
                raise Error("support_y labels must be non-negative")
            if not validate_pattern(support_x[support_idx], self.n_features):
                raise Error(
                    "support pattern must match n_features and contain finite"
                    " values"
                )
            var label_pos = self.find_label(support_y[support_idx])
            if label_pos == -1:
                self.labels.append(support_y[support_idx])
                var prototype = List[Float64]()
                for _ in range(self.n_features):
                    prototype.append(0.0)
                self.prototypes.append(prototype^)
                label_pos = len(self.labels) - 1
            for feature in range(self.n_features):
                self.prototypes[label_pos][feature] += support_x[support_idx][
                    feature
                ]
        for label_pos in range(len(self.labels)):
            var count = 0
            for support_idx in range(len(support_y)):
                if support_y[support_idx] == self.labels[label_pos]:
                    count += 1
            for feature in range(self.n_features):
                self.prototypes[label_pos][feature] = self.prototypes[
                    label_pos
                ][feature] / Float64(count)

    def find_label(self, label: Int) -> Int:
        for idx in range(len(self.labels)):
            if self.labels[idx] == label:
                return idx
        return -1

    def metric_score(
        self, query: List[Float64], prototype: List[Float64]
    ) -> Float64:
        if self.metric == 0:
            return cosine_score(query, prototype)
        if self.metric == 1:
            return euclidean_score(query, prototype)
        return hamming_score(query, prototype)


def vector3(a: Float64, b: Float64, c: Float64) -> List[Float64]:
    var values = List[Float64]()
    values.append(a)
    values.append(b)
    values.append(c)
    return values^


def vector4(a: Float64, b: Float64, c: Float64, d: Float64) -> List[Float64]:
    var values = List[Float64]()
    values.append(a)
    values.append(b)
    values.append(c)
    values.append(d)
    return values^


def validate_haam() raises -> Bool:
    var learner = HebbianFewShot(4, 2, 0.1)
    learner.store(vector4(1.0, 0.0, 0.0, 0.0), 0)
    if learner.query(vector4(0.9, 0.0, 0.0, 0.0)) != 0:
        return False

    var train = List[List[Float64]]()
    train.append(vector4(1.0, 0.0, 0.0, 0.0))
    train.append(vector4(0.0, 0.0, 0.0, 0.0))
    learner.reset()
    learner.store_temporal(train, 1)
    if learner.query(vector4(0.4, 0.0, 0.0, 0.0)) != 1:
        return False

    var support_x = List[List[Float64]]()
    support_x.append(vector3(1.0, 0.0, 0.0))
    support_x.append(vector3(0.0, 0.0, 1.0))
    var support_y = List[Int]()
    support_y.append(0)
    support_y.append(1)
    var query_x = List[List[Float64]]()
    query_x.append(vector3(0.8, 0.1, 0.0))
    for metric in range(3):
        var net = SpikePrototypeNet(3, metric)
        var predictions = net.classify(support_x, support_y, query_x)
        if predictions[0] != 0:
            return False
    return True


def main() raises:
    if not validate_haam():
        raise Error("HAAM validation failed")

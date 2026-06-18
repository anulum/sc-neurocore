# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Surrogate-guided stochastic-computing optimiser

"""Surrogate-guided SC compiler design-space search.

This module turns the existing analytical SC resource model into a trainable
compiler optimiser.  It fits a small ridge-regression surrogate over generated
candidate points plus optional measured benchmark observations, then chooses
per-layer bitstream length, decorrelator, mixed precision, and LFSR polynomial
under a target hardware budget.

The surrogate is deliberately small and deterministic: it gives the compiler a
learned ranking surface without adding heavyweight training dependencies or
claiming measured hardware results where only modelled estimates exist.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np

from sc_neurocore.optimizer.sc_optimizer import HardwareBudget, LayerProfile, SCOptimizer


_DECORRELATORS = ("None", "LFSR", "Sobol", "Halton", "SCC_Decorrelator")
_MODES = ("SC", "Hybrid", "Deterministic")
_LFSR_POLYNOMIALS = (
    "x16+x14+x13+x11+1",
    "x16+x15+x13+x4+1",
    "x16+x12+x3+x1+1",
)


@dataclass(frozen=True)
class TargetHardwareProfile:
    """Target device budget and compiler preference weights."""

    name: str
    budget: HardwareBudget
    lut_weight: float = 0.35
    power_weight: float = 0.35
    latency_weight: float = 0.20
    accuracy_weight: float = 1.0


@dataclass(frozen=True)
class BenchmarkObservation:
    """Measured or externally supplied design-point observation.

    The optimiser treats these as higher-priority training points than its
    analytical generated points.  Callers should only pass observations that
    come from real benchmark or synthesis outputs.
    """

    mac_count: int
    bitstream_length: int
    decorrelator: str
    mode: str
    precision_bits: int
    lfsr_polynomial: str
    luts_used: int
    power_mw: float
    latency_cycles: int
    accuracy_score: float
    is_critical_path: bool = False


@dataclass(frozen=True)
class SurrogateLayerConfig:
    """Selected SC compiler settings for one layer."""

    bitstream_length: int
    decorrelator: str
    mode: str
    precision_bits: int
    lfsr_polynomial: str
    luts_used: int
    power_used: float
    latency_cycles: int
    accuracy_score: float
    utility_score: float


@dataclass(frozen=True)
class SurrogateOptimizerReport:
    """Budgeted per-layer compiler configuration."""

    config: dict[str, SurrogateLayerConfig]
    total_luts: int
    total_power_mw: float
    total_latency_cycles: int
    mean_accuracy: float
    training_points: int
    target_name: str
    rejected_layers: list[str] = field(default_factory=list)

    @property
    def feasible(self) -> bool:
        """Whether every layer received a configuration."""
        return not self.rejected_layers


@dataclass(frozen=True)
class _Candidate:
    mac_count: int
    is_critical_path: bool
    bitstream_length: int
    decorrelator: str
    mode: str
    precision_bits: int
    lfsr_polynomial: str


@dataclass(frozen=True)
class _Label:
    luts: float
    power_mw: float
    latency_cycles: float
    accuracy: float


class _RidgeSurrogate:
    """Small multi-output ridge regressor backed by NumPy."""

    def __init__(self, alpha: float = 1e-3) -> None:
        self.alpha = alpha
        self._coef: np.ndarray[Any, Any] | None = None

    def fit(self, features: np.ndarray[Any, Any], labels: np.ndarray[Any, Any]) -> None:
        if features.ndim != 2 or labels.ndim != 2:
            raise ValueError("features and labels must be two-dimensional")
        if features.shape[0] != labels.shape[0]:
            raise ValueError("feature/label row count mismatch")
        reg = self.alpha * np.eye(features.shape[1], dtype=np.float64)
        self._coef = np.linalg.solve(features.T @ features + reg, features.T @ labels)

    def predict(self, features: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        if self._coef is None:
            raise RuntimeError("surrogate is not fitted")
        prediction: np.ndarray[Any, Any] = features @ self._coef
        return prediction


class SurrogateSCOptimizer:
    """Compiler optimiser using a learned surrogate over SC design points."""

    def __init__(
        self,
        target: TargetHardwareProfile,
        *,
        bitstream_options: Iterable[int] = (64, 128, 256, 512, 1024, 2048),
        precision_options: Iterable[int] = (4, 6, 8, 12, 16),
        lfsr_polynomials: Iterable[str] = _LFSR_POLYNOMIALS,
        observations: Iterable[BenchmarkObservation] = (),
    ) -> None:
        self.target = target
        self.bitstream_options = tuple(sorted({int(v) for v in bitstream_options}))
        self.precision_options = tuple(sorted({int(v) for v in precision_options}))
        self.lfsr_polynomials = tuple(lfsr_polynomials)
        self.observations = tuple(observations)
        self._base = SCOptimizer(target.budget)
        self._surrogate = _RidgeSurrogate()
        self._training_points = 0

    def optimise(self, network: list[LayerProfile]) -> SurrogateOptimizerReport | None:
        """Select budgeted layer settings for ``network``."""
        if not network:
            return SurrogateOptimizerReport(
                config={},
                total_luts=0,
                total_power_mw=0.0,
                total_latency_cycles=0,
                mean_accuracy=0.0,
                training_points=0,
                target_name=self.target.name,
            )

        self._fit_surrogate(network)
        selected: dict[str, SurrogateLayerConfig] = {}
        rejected: list[str] = []

        for layer in sorted(network, key=lambda item: item.is_critical_path, reverse=True):
            remaining_luts = self.target.budget.max_luts - sum(
                c.luts_used for c in selected.values()
            )
            remaining_power = self.target.budget.max_power_mw - sum(
                c.power_used for c in selected.values()
            )
            candidates = self._rank_layer_candidates(layer, remaining_luts, remaining_power)
            if not candidates:
                rejected.append(layer.id)
                continue
            selected[layer.id] = candidates[0]

        if rejected:
            return SurrogateOptimizerReport(
                config=selected,
                total_luts=sum(c.luts_used for c in selected.values()),
                total_power_mw=sum(c.power_used for c in selected.values()),
                total_latency_cycles=max((c.latency_cycles for c in selected.values()), default=0),
                mean_accuracy=self._weighted_accuracy(selected, network),
                training_points=self._training_points,
                target_name=self.target.name,
                rejected_layers=rejected,
            )

        return self._rebalance(selected, network)

    def _fit_surrogate(self, network: list[LayerProfile]) -> None:
        rows: list[np.ndarray[Any, Any]] = []
        labels: list[np.ndarray[Any, Any]] = []

        for layer in network:
            for cand in self._candidate_grid(layer):
                label = self._analytical_label(cand)
                rows.append(self._features(cand))
                labels.append(self._normalise_label(label))

        for obs in self.observations:
            cand = _Candidate(
                mac_count=obs.mac_count,
                is_critical_path=obs.is_critical_path,
                bitstream_length=obs.bitstream_length,
                decorrelator=obs.decorrelator,
                mode=obs.mode,
                precision_bits=obs.precision_bits,
                lfsr_polynomial=obs.lfsr_polynomial,
            )
            label = _Label(
                luts=float(obs.luts_used),
                power_mw=obs.power_mw,
                latency_cycles=float(obs.latency_cycles),
                accuracy=obs.accuracy_score,
            )
            rows.extend([self._features(cand)] * 4)
            labels.extend([self._normalise_label(label)] * 4)

        self._training_points = len(rows)
        self._surrogate.fit(np.vstack(rows), np.vstack(labels))

    def _rank_layer_candidates(
        self, layer: LayerProfile, remaining_luts: int, remaining_power: float
    ) -> list[SurrogateLayerConfig]:
        ranked: list[SurrogateLayerConfig] = []
        for cand in self._candidate_grid(layer):
            pred = self._denormalise_prediction(
                self._surrogate.predict(self._features(cand)[None, :])[0]
            )
            pred = self._observation_label(cand) or pred
            if pred.luts > remaining_luts or pred.power_mw > remaining_power:
                continue
            if self.target.budget.max_latency_cycles and (
                pred.latency_cycles > self.target.budget.max_latency_cycles
            ):
                continue
            ranked.append(self._to_config(cand, pred))
        ranked.sort(key=lambda cfg: cfg.utility_score, reverse=True)
        return ranked

    def _candidate_grid(self, layer: LayerProfile) -> Iterable[_Candidate]:
        for mode in _MODES:
            if mode == "Deterministic":
                yield _Candidate(
                    mac_count=layer.mac_count,
                    is_critical_path=layer.is_critical_path,
                    bitstream_length=1,
                    decorrelator="None",
                    mode=mode,
                    precision_bits=16,
                    lfsr_polynomial="none",
                )
                continue
            for length in self.bitstream_options:
                for precision in self.precision_options:
                    for decorrelator in _DECORRELATORS:
                        polys = self.lfsr_polynomials if decorrelator == "LFSR" else ("none",)
                        for polynomial in polys:
                            yield _Candidate(
                                mac_count=layer.mac_count,
                                is_critical_path=layer.is_critical_path,
                                bitstream_length=length,
                                decorrelator=decorrelator,
                                mode=mode,
                                precision_bits=precision,
                                lfsr_polynomial=polynomial,
                            )

    def _analytical_label(self, cand: _Candidate) -> _Label:
        luts, power, accuracy, latency = self._base._estimate_resources(
            cand.mac_count,
            cand.bitstream_length,
            cand.decorrelator,
            cand.mode,
        )
        precision_scale = cand.precision_bits / 16.0
        if cand.mode != "Deterministic":
            luts = int(luts * (0.70 + 0.30 * precision_scale))
            power = power * (0.60 + 0.40 * precision_scale)
            accuracy -= max(0.0, (8 - cand.precision_bits) * 0.012)
            accuracy += self._polynomial_quality(cand.lfsr_polynomial)
        return _Label(
            luts=max(1.0, float(luts)),
            power_mw=max(1e-9, float(power)),
            latency_cycles=max(1.0, float(latency)),
            accuracy=max(0.1, min(1.0, float(accuracy))),
        )

    def _observation_label(self, cand: _Candidate) -> _Label | None:
        for obs in self.observations:
            if (
                obs.mac_count == cand.mac_count
                and obs.is_critical_path == cand.is_critical_path
                and obs.bitstream_length == cand.bitstream_length
                and obs.decorrelator == cand.decorrelator
                and obs.mode == cand.mode
                and obs.precision_bits == cand.precision_bits
                and obs.lfsr_polynomial == cand.lfsr_polynomial
            ):
                return _Label(
                    luts=float(obs.luts_used),
                    power_mw=obs.power_mw,
                    latency_cycles=float(obs.latency_cycles),
                    accuracy=obs.accuracy_score,
                )
        return None

    def _to_config(self, cand: _Candidate, pred: _Label) -> SurrogateLayerConfig:
        return SurrogateLayerConfig(
            bitstream_length=cand.bitstream_length,
            decorrelator=cand.decorrelator,
            mode=cand.mode,
            precision_bits=cand.precision_bits,
            lfsr_polynomial=cand.lfsr_polynomial,
            luts_used=max(1, int(round(pred.luts))),
            power_used=max(0.0, pred.power_mw),
            latency_cycles=max(1, int(round(pred.latency_cycles))),
            accuracy_score=max(0.0, min(1.0, pred.accuracy)),
            utility_score=self._utility(pred),
        )

    def _rebalance(
        self, selected: dict[str, SurrogateLayerConfig], network: list[LayerProfile]
    ) -> SurrogateOptimizerReport:
        # Greedy second pass: upgrade the most useful affordable candidate,
        # especially on critical layers, until no candidate improves utility.
        improved = True
        while improved:
            improved = False
            best_layer = ""
            best_cfg: SurrogateLayerConfig | None = None
            best_gain = 0.0
            current_luts = sum(c.luts_used for c in selected.values())
            current_power = sum(c.power_used for c in selected.values())

            for layer in network:
                current = selected[layer.id]
                for cand in self._rank_layer_candidates(
                    layer,
                    self.target.budget.max_luts - current_luts + current.luts_used,
                    self.target.budget.max_power_mw - current_power + current.power_used,
                ):
                    if cand == current:
                        continue
                    gain = cand.utility_score - current.utility_score
                    if layer.is_critical_path:
                        gain *= 1.5
                    if gain > best_gain:
                        best_gain = gain
                        best_cfg = cand
                        best_layer = layer.id

            if best_cfg is not None and best_layer:
                selected[best_layer] = best_cfg
                improved = True

        return SurrogateOptimizerReport(
            config=selected,
            total_luts=sum(c.luts_used for c in selected.values()),
            total_power_mw=sum(c.power_used for c in selected.values()),
            total_latency_cycles=max((c.latency_cycles for c in selected.values()), default=0),
            mean_accuracy=self._weighted_accuracy(selected, network),
            training_points=self._training_points,
            target_name=self.target.name,
        )

    def _features(self, cand: _Candidate) -> np.ndarray[Any, Any]:
        decor = [1.0 if cand.decorrelator == name else 0.0 for name in _DECORRELATORS]
        mode = [1.0 if cand.mode == name else 0.0 for name in _MODES]
        poly = (
            0.0
            if cand.lfsr_polynomial == "none"
            else self._polynomial_quality(cand.lfsr_polynomial)
        )
        return np.array(
            [
                1.0,
                math.log2(max(1, cand.mac_count)),
                math.log2(max(1, cand.bitstream_length)),
                cand.precision_bits / 16.0,
                1.0 if cand.is_critical_path else 0.0,
                poly,
                *decor,
                *mode,
            ],
            dtype=np.float64,
        )

    def _normalise_label(self, label: _Label) -> np.ndarray[Any, Any]:
        return np.array(
            [
                label.luts / max(1, self.target.budget.max_luts),
                label.power_mw / max(1e-9, self.target.budget.max_power_mw),
                label.latency_cycles / max(1, self.target.budget.max_latency_cycles or 2048),
                label.accuracy,
            ],
            dtype=np.float64,
        )

    def _denormalise_prediction(self, values: np.ndarray[Any, Any]) -> _Label:
        return _Label(
            luts=float(values[0] * max(1, self.target.budget.max_luts)),
            power_mw=float(values[1] * max(1e-9, self.target.budget.max_power_mw)),
            latency_cycles=float(values[2] * max(1, self.target.budget.max_latency_cycles or 2048)),
            accuracy=float(values[3]),
        )

    def _utility(self, label: _Label) -> float:
        lut_frac = label.luts / max(1, self.target.budget.max_luts)
        power_frac = label.power_mw / max(1e-9, self.target.budget.max_power_mw)
        latency_frac = label.latency_cycles / max(1, self.target.budget.max_latency_cycles or 2048)
        return (
            self.target.accuracy_weight * label.accuracy
            - self.target.lut_weight * lut_frac
            - self.target.power_weight * power_frac
            - self.target.latency_weight * latency_frac
        )

    @staticmethod
    def _polynomial_quality(polynomial: str) -> float:
        if polynomial == "x16+x14+x13+x11+1":
            return 0.006
        if polynomial == "x16+x15+x13+x4+1":
            return 0.003
        if polynomial == "x16+x12+x3+x1+1":
            return 0.001
        return 0.0

    @staticmethod
    def _weighted_accuracy(
        selected: dict[str, SurrogateLayerConfig], network: list[LayerProfile]
    ) -> float:
        total = 0.0
        weight = 0.0
        for layer in network:
            cfg = selected.get(layer.id)
            if cfg is None:
                continue
            w = 2.0 if layer.is_critical_path else 1.0
            total += cfg.accuracy_score * w
            weight += w
        return total / weight if weight else 0.0

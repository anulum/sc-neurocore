# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Closed-Loop Adaptive Controller

"""Closed-loop controller bridging SC-Runtime drift detection
to SC-Optimizer re-optimisation.

When the runtime observes drift (SCC or density shift), this controller
triggers an SA re-optimisation and produces a new RuntimeConfig.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, List, Optional

import numpy as np

from sc_neurocore.core.types import HardwareBudget, LayerSpec
from sc_neurocore.optimizer.sc_optimizer import (
    SCOptimizer,
    HardwareBudget as OptBudget,
    LayerProfile,
    OptimizerReport,
)
from sc_neurocore.control.sc_runtime import (
    ActivityMonitor,
    RuntimeConfig,
)


@dataclass
class AdaptationEvent:
    """Record of a single adaptation cycle."""

    timestamp: float
    trigger_reason: str
    old_accuracy: float
    new_accuracy: float
    elapsed_ms: float
    config_changed: bool


@dataclass
class AdaptiveLoopConfig:
    """Configuration for the adaptive controller."""

    drift_threshold: float = 0.3
    reoptimize_cooldown_s: float = 1.0
    sa_max_iter: int = 500
    sa_seed: int = 42
    enable_logging: bool = True


class AdaptiveController:
    """Closed-loop: Runtime drift → Optimizer SA → New config.

    Usage::

        ctrl = AdaptiveController(budget, layers)
        for bitstream_pair in stream:
            event = ctrl.step(bitstream_pair)
            if event and event.config_changed:
                apply_new_config(ctrl.current_config)
    """

    def __init__(
        self,
        budget: HardwareBudget,
        layers: List[LayerSpec],
        config: AdaptiveLoopConfig | None = None,
    ):
        self.budget = budget
        self.layers = layers
        self.config = config or AdaptiveLoopConfig()
        self.monitor = ActivityMonitor(
            window_size=100,
            drift_threshold=self.config.drift_threshold,
        )
        self._opt_budget = OptBudget(
            max_luts=budget.max_luts,
            max_power_mw=budget.max_power_mw,
            max_latency_cycles=budget.max_latency_cycles,
        )
        self.optimizer = SCOptimizer(self._opt_budget)
        self.current_report: Optional[OptimizerReport] = None
        self.current_config: Optional[RuntimeConfig] = None
        self.adaptation_log: List[AdaptationEvent] = []
        self._last_reopt_time: float = 0.0

    def step(
        self,
        bitstream_a: np.ndarray[Any, Any],
        bitstream_b: np.ndarray[Any, Any],
    ) -> Optional[AdaptationEvent]:
        """Feed a bitstream pair; returns AdaptationEvent if re-optimisation triggered."""
        self.monitor.observe(bitstream_a, bitstream_b)

        if not self.monitor.drift_active:
            return None

        now = time.monotonic()
        if now - self._last_reopt_time < self.config.reoptimize_cooldown_s:
            return None

        old_accuracy = self.current_report.mean_accuracy if self.current_report else 0.0

        network = [
            LayerProfile(
                id=ls.layer_id,
                mac_count=max(ls.mac_count, ls.neurons),
                is_critical_path=ls.is_critical_path,
            )
            for ls in self.layers
        ]

        t0 = time.perf_counter()
        report = self.optimizer.optimize_annealing(
            network,
            max_iter=self.config.sa_max_iter,
            seed=self.config.sa_seed,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000

        config_changed = report is not None
        new_accuracy = report.mean_accuracy if report else old_accuracy

        if report:
            self.current_report = report
            best_layer = max(report.config.values(), key=lambda c: c.accuracy_score)
            self.current_config = RuntimeConfig(
                bitstream_length=best_layer.bitstream_length or 256,
            )

        event = AdaptationEvent(
            timestamp=now,
            trigger_reason=f"drift_scc={self.monitor.mean_scc:.3f}",
            old_accuracy=old_accuracy,
            new_accuracy=new_accuracy,
            elapsed_ms=elapsed_ms,
            config_changed=config_changed,
        )
        self.adaptation_log.append(event)
        self._last_reopt_time = now
        return event

    @property
    def adaptation_rate(self) -> float:
        """Fraction of steps that triggered re-optimisation."""
        n = self.monitor._step_count if hasattr(self.monitor, "_step_count") else 1
        return len(self.adaptation_log) / max(n, 1)

    def summary(self) -> str:
        lines = [
            f"AdaptiveController: {len(self.adaptation_log)} adaptations",
            f"  Current accuracy: {self.current_report.mean_accuracy:.4f}"
            if self.current_report
            else "  No optimisation yet",
            f"  Drift active: {self.monitor.drift_active}",
            f"  Mean SCC: {self.monitor.mean_scc:.4f}",
        ]
        return "\n".join(lines)

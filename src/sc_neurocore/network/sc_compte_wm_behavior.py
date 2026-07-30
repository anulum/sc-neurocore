# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

"""Ensemble behavior contract for the SC Compte working-memory ring.

The protocol separates spontaneous baseline, cue, delay, distractor,
recovery, response, and reset epochs. Its circular acceptance statistics turn
the complete 2,560-cell execution into an explicit SC-network claim without
altering or relabelling the preserved source-bounded scalar neuron.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from .sc_compte_wm import SCCompteWMNetworkSpec, circular_displacement_deg
from .sc_compte_wm_backends import (
    SCCompteWMBackend,
    SCCompteWMBackendRun,
    run_sc_compte_wm_network,
)
from .sc_compte_wm_network import SCCompteWMStimulus

SC_COMPTE_WM_BEHAVIOR_BACKENDS: tuple[SCCompteWMBackend, ...] = (
    "python",
    "rust",
    "julia",
    "go",
    "mojo",
)
SC_COMPTE_WM_BEHAVIOR_REFERENCE_SEEDS: tuple[int, ...] = (41, 42, 43)


@dataclass(frozen=True, slots=True)
class SCCompteWMBehaviorProtocol:
    """Frozen 2.5-second SC working-memory behavior protocol in milliseconds."""

    cue_center_deg: float = 180.0
    distractor_center_deg: float = 270.0
    window_ms: float = 250.0
    duration_ms: float = 2500.0
    cue_start_ms: float = 250.0
    cue_duration_ms: float = 250.0
    cue_current_pa: float = 200.0
    distractor_start_ms: float = 1000.0
    distractor_duration_ms: float = 250.0
    distractor_current_pa: float = 200.0
    response_start_ms: float = 1750.0
    response_duration_ms: float = 250.0
    response_current_pa: float = 500.0

    def __post_init__(self) -> None:
        values = (
            self.cue_center_deg,
            self.distractor_center_deg,
            self.window_ms,
            self.duration_ms,
            self.cue_start_ms,
            self.cue_duration_ms,
            self.cue_current_pa,
            self.distractor_start_ms,
            self.distractor_duration_ms,
            self.distractor_current_pa,
            self.response_start_ms,
            self.response_duration_ms,
            self.response_current_pa,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("behavior protocol values must be finite")
        if any(
            value <= 0.0
            for value in (
                self.window_ms,
                self.duration_ms,
                self.cue_duration_ms,
                self.cue_current_pa,
                self.distractor_duration_ms,
                self.distractor_current_pa,
                self.response_duration_ms,
                self.response_current_pa,
            )
        ):
            raise ValueError("behavior durations, window, and currents must be positive")
        if any(value < 0.0 for value in self.epoch_starts_ms):
            raise ValueError("behavior epoch starts must be non-negative")
        if self.duration_ms / self.window_ms != 10.0:
            raise ValueError("behavior v1 requires exactly ten statistics windows")
        epochs = (
            (self.cue_start_ms, self.cue_duration_ms),
            (self.distractor_start_ms, self.distractor_duration_ms),
            (self.response_start_ms, self.response_duration_ms),
        )
        if any(start + duration > self.duration_ms for start, duration in epochs):
            raise ValueError("behavior epochs must remain inside the run")

    @property
    def epoch_starts_ms(self) -> tuple[float, float, float]:
        """Return cue, distractor, and response start times."""
        return self.cue_start_ms, self.distractor_start_ms, self.response_start_ms

    def stimuli(self) -> tuple[SCCompteWMStimulus, ...]:
        """Return the localized cue/distractor and global response stimuli."""
        return (
            SCCompteWMStimulus(
                self.cue_start_ms,
                self.cue_duration_ms,
                self.cue_current_pa,
                center_deg=self.cue_center_deg,
            ),
            SCCompteWMStimulus(
                self.distractor_start_ms,
                self.distractor_duration_ms,
                self.distractor_current_pa,
                center_deg=self.distractor_center_deg,
            ),
            SCCompteWMStimulus(
                self.response_start_ms,
                self.response_duration_ms,
                self.response_current_pa,
                kind="global_current",
                center_deg=None,
            ),
        )


@dataclass(frozen=True, slots=True)
class SCCompteWMBehaviorAcceptance:
    """Predeclared v1 thresholds for classifying one protocol receipt."""

    maximum_baseline_rate_hz: float = 1.0
    minimum_cue_rate_hz: float = 3.0
    minimum_cue_resultant: float = 0.75
    maximum_cue_error_deg: float = 10.0
    minimum_delay_rate_hz: float = 6.0
    minimum_delay_resultant: float = 0.80
    maximum_delay_cue_error_deg: float = 15.0
    maximum_delay_drift_deg: float = 10.0
    minimum_recovery_resultant: float = 0.70
    maximum_recovery_cue_error_deg: float = 45.0
    minimum_response_rate_hz: float = 50.0
    maximum_response_resultant: float = 0.10
    maximum_reset_rate_hz: float = 5.0
    maximum_reset_resultant: float = 0.20
    maximum_ensemble_mean_drift_deg: float = 5.0


@dataclass(frozen=True, slots=True)
class SCCompteWMBehaviorMetrics:
    """Circular and rate observables extracted from one ten-window receipt."""

    baseline_rate_hz: float
    cue_rate_hz: float
    cue_resultant: float
    cue_error_deg: float
    minimum_delay_rate_hz: float
    minimum_delay_resultant: float
    maximum_delay_cue_error_deg: float
    signed_delay_drift_deg: float
    recovery_rate_hz: float
    recovery_resultant: float
    recovery_cue_error_deg: float
    recovery_distractor_error_deg: float
    distractor_resistance_margin_deg: float
    response_rate_hz: float
    response_resultant: float
    reset_rate_hz: float
    reset_resultant: float


@dataclass(frozen=True, slots=True)
class SCCompteWMBehaviorTrial:
    """One selected-runtime, one-seed behavior classification and custody."""

    backend: SCCompteWMBackend
    seed: int
    execution_ns: int
    input_sha256: str
    spike_sha256: str
    final_state_sha256: str
    excitatory_spikes: int
    inhibitory_spikes: int
    metrics: SCCompteWMBehaviorMetrics
    checks: tuple[tuple[str, bool], ...]
    passed: bool


@dataclass(frozen=True, slots=True)
class SCCompteWMBehaviorEnsemble:
    """Aggregate acceptance for the reference seeds and all runtime anchors."""

    reference_backend: SCCompteWMBackend
    reference_seeds: tuple[int, ...]
    anchor_seed: int
    represented_backends: tuple[SCCompteWMBackend, ...]
    all_runtime_input_spike_count_exact: bool
    signed_delay_drifts_deg: tuple[float, ...]
    mean_signed_delay_drift_deg: float
    bidirectional_seed_drift: bool
    all_trials_passed: bool
    passed: bool


_DEFAULT_PROTOCOL = SCCompteWMBehaviorProtocol()
_DEFAULT_ACCEPTANCE = SCCompteWMBehaviorAcceptance()


def _angle_error_deg(observed_deg: float, target_deg: float) -> float:
    return abs(circular_displacement_deg(target_deg, observed_deg))


def assess_sc_compte_wm_behavior(
    run: SCCompteWMBackendRun,
    *,
    protocol: SCCompteWMBehaviorProtocol = _DEFAULT_PROTOCOL,
    acceptance: SCCompteWMBehaviorAcceptance = _DEFAULT_ACCEPTANCE,
) -> SCCompteWMBehaviorTrial:
    """Classify one complete backend receipt against the frozen v1 protocol."""
    receipt = run.receipt
    if receipt.duration_ms != protocol.duration_ms or len(receipt.windows) != 10:
        raise ValueError("behavior receipt must contain the frozen ten-window run")
    statistics = tuple(window.statistics for window in receipt.windows)
    if any(value is None for value in statistics):
        raise ValueError("every behavior window must contain excitatory statistics")
    stats = tuple(value for value in statistics if value is not None)
    baseline, cue = stats[0], stats[1]
    delay = stats[2:4]
    recovery = stats[6]
    response = stats[7]
    reset = stats[9]
    cue_error = _angle_error_deg(cue.bump_angle_deg, protocol.cue_center_deg)
    delay_errors = tuple(
        _angle_error_deg(value.bump_angle_deg, protocol.cue_center_deg) for value in delay
    )
    signed_drift = circular_displacement_deg(delay[0].bump_angle_deg, delay[-1].bump_angle_deg)
    recovery_cue_error = _angle_error_deg(recovery.bump_angle_deg, protocol.cue_center_deg)
    recovery_distractor_error = _angle_error_deg(
        recovery.bump_angle_deg, protocol.distractor_center_deg
    )
    metrics = SCCompteWMBehaviorMetrics(
        baseline_rate_hz=baseline.excitatory_rate_hz,
        cue_rate_hz=cue.excitatory_rate_hz,
        cue_resultant=cue.resultant_length,
        cue_error_deg=cue_error,
        minimum_delay_rate_hz=min(value.excitatory_rate_hz for value in delay),
        minimum_delay_resultant=min(value.resultant_length for value in delay),
        maximum_delay_cue_error_deg=max(delay_errors),
        signed_delay_drift_deg=signed_drift,
        recovery_rate_hz=recovery.excitatory_rate_hz,
        recovery_resultant=recovery.resultant_length,
        recovery_cue_error_deg=recovery_cue_error,
        recovery_distractor_error_deg=recovery_distractor_error,
        distractor_resistance_margin_deg=recovery_distractor_error - recovery_cue_error,
        response_rate_hz=response.excitatory_rate_hz,
        response_resultant=response.resultant_length,
        reset_rate_hz=reset.excitatory_rate_hz,
        reset_resultant=reset.resultant_length,
    )
    checks = (
        ("spontaneous_baseline", metrics.baseline_rate_hz <= acceptance.maximum_baseline_rate_hz),
        (
            "cue_formation",
            metrics.cue_rate_hz >= acceptance.minimum_cue_rate_hz
            and metrics.cue_resultant >= acceptance.minimum_cue_resultant
            and metrics.cue_error_deg <= acceptance.maximum_cue_error_deg,
        ),
        (
            "delay_persistence",
            metrics.minimum_delay_rate_hz >= acceptance.minimum_delay_rate_hz
            and metrics.minimum_delay_resultant >= acceptance.minimum_delay_resultant
            and metrics.maximum_delay_cue_error_deg <= acceptance.maximum_delay_cue_error_deg,
        ),
        (
            "bounded_delay_drift",
            abs(metrics.signed_delay_drift_deg) <= acceptance.maximum_delay_drift_deg,
        ),
        (
            "distractor_resistance",
            metrics.recovery_resultant >= acceptance.minimum_recovery_resultant
            and metrics.recovery_cue_error_deg <= acceptance.maximum_recovery_cue_error_deg
            and metrics.distractor_resistance_margin_deg > 0.0,
        ),
        (
            "global_response",
            metrics.response_rate_hz >= acceptance.minimum_response_rate_hz
            and metrics.response_resultant <= acceptance.maximum_response_resultant,
        ),
        (
            "response_reset",
            metrics.reset_rate_hz <= acceptance.maximum_reset_rate_hz
            and metrics.reset_resultant <= acceptance.maximum_reset_resultant,
        ),
    )
    return SCCompteWMBehaviorTrial(
        backend=run.backend,
        seed=receipt.seed,
        execution_ns=run.execution_ns,
        input_sha256=receipt.input_sha256,
        spike_sha256=receipt.spike_sha256,
        final_state_sha256=receipt.final_state_sha256,
        excitatory_spikes=receipt.excitatory_spikes,
        inhibitory_spikes=receipt.inhibitory_spikes,
        metrics=metrics,
        checks=checks,
        passed=all(passed for _, passed in checks),
    )


def run_sc_compte_wm_behavior_trial(
    *,
    backend: SCCompteWMBackend,
    seed: int,
    protocol: SCCompteWMBehaviorProtocol = _DEFAULT_PROTOCOL,
    acceptance: SCCompteWMBehaviorAcceptance = _DEFAULT_ACCEPTANCE,
    timeout_s: float | None = None,
) -> SCCompteWMBehaviorTrial:
    """Execute and classify one modulated-network behavior trial."""
    run = run_sc_compte_wm_network(
        protocol.duration_ms,
        backend=backend,
        spec=SCCompteWMNetworkSpec(seed=seed, modulated=True),
        stimuli=protocol.stimuli(),
        statistics_window_ms=protocol.window_ms,
        timeout_s=timeout_s,
    )
    return assess_sc_compte_wm_behavior(run, protocol=protocol, acceptance=acceptance)


def summarize_sc_compte_wm_behavior_ensemble(
    trials: tuple[SCCompteWMBehaviorTrial, ...],
    *,
    reference_backend: SCCompteWMBackend = "rust",
    reference_seeds: tuple[int, ...] = SC_COMPTE_WM_BEHAVIOR_REFERENCE_SEEDS,
    required_backends: tuple[SCCompteWMBackend, ...] = SC_COMPTE_WM_BEHAVIOR_BACKENDS,
    anchor_seed: int = 42,
    acceptance: SCCompteWMBehaviorAcceptance = _DEFAULT_ACCEPTANCE,
) -> SCCompteWMBehaviorEnsemble:
    """Require three reference seeds, bidirectional drift, and exact route anchors."""
    if not trials:
        raise ValueError("behavior ensemble requires at least one trial")
    references = {
        trial.seed: trial
        for trial in trials
        if trial.backend == reference_backend and trial.seed in reference_seeds
    }
    if tuple(sorted(references)) != tuple(sorted(reference_seeds)):
        raise ValueError("behavior ensemble is missing a required reference seed")
    represented = tuple(
        backend
        for backend in required_backends
        if any(trial.backend == backend for trial in trials)
    )
    anchors = {
        trial.backend: trial
        for trial in trials
        if trial.seed == anchor_seed and trial.backend in required_backends
    }
    anchor_custody = {
        (
            trial.input_sha256,
            trial.spike_sha256,
            trial.excitatory_spikes,
            trial.inhibitory_spikes,
        )
        for trial in anchors.values()
    }
    exact_anchor_custody = len(anchors) == len(required_backends) and len(anchor_custody) == 1
    drifts = tuple(references[seed].metrics.signed_delay_drift_deg for seed in reference_seeds)
    mean_drift = sum(drifts) / len(drifts)
    bidirectional = min(drifts) < 0.0 < max(drifts)
    all_trials_passed = all(trial.passed for trial in trials)
    passed = (
        represented == required_backends
        and exact_anchor_custody
        and bidirectional
        and abs(mean_drift) <= acceptance.maximum_ensemble_mean_drift_deg
        and all_trials_passed
    )
    return SCCompteWMBehaviorEnsemble(
        reference_backend=reference_backend,
        reference_seeds=reference_seeds,
        anchor_seed=anchor_seed,
        represented_backends=represented,
        all_runtime_input_spike_count_exact=exact_anchor_custody,
        signed_delay_drifts_deg=drifts,
        mean_signed_delay_drift_deg=mean_drift,
        bidirectional_seed_drift=bidirectional,
        all_trials_passed=all_trials_passed,
        passed=passed,
    )


__all__ = [
    "SC_COMPTE_WM_BEHAVIOR_BACKENDS",
    "SC_COMPTE_WM_BEHAVIOR_REFERENCE_SEEDS",
    "SCCompteWMBehaviorAcceptance",
    "SCCompteWMBehaviorEnsemble",
    "SCCompteWMBehaviorMetrics",
    "SCCompteWMBehaviorProtocol",
    "SCCompteWMBehaviorTrial",
    "assess_sc_compte_wm_behavior",
    "run_sc_compte_wm_behavior_trial",
    "summarize_sc_compte_wm_behavior_ensemble",
]

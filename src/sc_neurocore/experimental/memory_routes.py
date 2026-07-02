# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Bounded memory experiments for the safe alternative-path harness

"""Bounded-memory experiment routes for the safe alternative-path harness.

Registers baseline/candidate route pairs that compare full-history buffers
against fixed-footprint streaming estimators under the alternative-path harness.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from sc_neurocore.neurons.stochastic_lif import StochasticLIFNeuron

from .alternative_path import AlternativePathConfig, AlternativePathRoute, ComparisonStats

_DEFAULT_CUES = np.array(
    [
        [1.0, 1.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 1.0, 0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0, 1.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 1.0, 1.0, 0.0],
    ],
    dtype=np.float64,
)


def _validated_delay_steps(delay_steps: int) -> int:
    """Return a simulation delay after rejecting bool or negative values."""
    if isinstance(delay_steps, bool) or delay_steps < 0:
        raise ValueError("delay_steps must be non-negative")
    return int(delay_steps)


def _validated_positive_count(name: str, value: int) -> int:
    """Return a strictly positive integer count for route dimensions or seeds."""
    if isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be positive")
    return int(value)


def _validated_cue_matrix(cues: np.ndarray[Any, Any] | None) -> npt.NDArray[np.float64]:
    """Return a finite binary two-dimensional cue matrix for recall trials."""
    cue_matrix = np.asarray(cues if cues is not None else _DEFAULT_CUES, dtype=np.float64)
    if cue_matrix.ndim != 2:
        raise ValueError("cues must be a finite 2D matrix")
    if cue_matrix.shape[0] == 0 or cue_matrix.shape[1] == 0:
        raise ValueError("cues must contain at least one cue and one neuron")
    if not np.all(np.isfinite(cue_matrix)):
        raise ValueError("cues must be finite")
    if not np.all((cue_matrix == 0.0) | (cue_matrix == 1.0)):
        raise ValueError("cues must contain only binary 0.0 or 1.0 values")
    return cue_matrix


def _make_memory_neurons(n_neurons: int, seed: int) -> list[StochasticLIFNeuron]:
    return [
        StochasticLIFNeuron(
            v_rest=0.0,
            v_reset=0.0,
            v_threshold=1.0,
            tau_mem=4.0,
            dt=1.0,
            noise_std=0.0,
            resistance=1.0,
            refractory_period=0,
            seed=seed * 31 + index,
        )
        for index in range(n_neurons)
    ]


def _run_delayed_recall_trial(
    cue: np.ndarray[Any, Any],
    *,
    delay_steps: int,
    write_matrix: np.ndarray[Any, Any] | None,
    read_matrix: np.ndarray[Any, Any] | None,
    encode_steps: int = 4,
    recall_steps: int = 4,
    encode_current: float = 1.3,
    local_encode_decay: float = 0.6,
    local_delay_decay: float = 0.55,
    local_recall_decay: float = 0.65,
    shared_encode_decay: float = 0.94,
    shared_delay_decay: float = 0.985,
    shared_recall_decay: float = 0.99,
    shared_encode_gain: float = 0.55,
    shared_delay_gain: float = 0.35,
    shared_recall_gain: float = 0.7,
    shared_write_gain_delay: float = 0.25,
    shared_write_gain_recall: float = 0.15,
    local_spike_gain_delay: float = 0.2,
    local_trace_gain_recall: float = 0.6,
    local_spike_gain_recall: float = 0.5,
    seed: int = 0,
) -> tuple[float, np.ndarray[Any, Any]]:
    cue_arr = np.asarray(cue, dtype=np.float64).reshape(-1)
    neurons = _make_memory_neurons(cue_arr.size, seed)
    local_trace: npt.NDArray[np.float64] = np.zeros(cue_arr.size, dtype=np.float64)
    use_shared_state = write_matrix is not None and read_matrix is not None
    if use_shared_state:
        assert write_matrix is not None
        assert read_matrix is not None
        shared_state: npt.NDArray[np.float64] = np.zeros(write_matrix.shape[0], dtype=np.float64)
    else:
        shared_state = np.zeros(0, dtype=np.float64)

    for _ in range(encode_steps):
        if use_shared_state:
            assert read_matrix is not None
            feedback = (read_matrix @ shared_state) * shared_encode_gain
        else:
            feedback = np.zeros_like(cue_arr)
        spikes = np.array(
            [
                neuron.step(float(encode_current * cue_arr[index] + feedback[index]))
                for index, neuron in enumerate(neurons)
            ],
            dtype=np.float64,
        )
        local_trace = local_encode_decay * local_trace + spikes
        if use_shared_state:
            assert write_matrix is not None
            shared_state = shared_encode_decay * shared_state + write_matrix @ spikes

    for _ in range(delay_steps):
        if use_shared_state:
            assert read_matrix is not None
            feedback = (read_matrix @ shared_state) * shared_delay_gain
        else:
            feedback = np.zeros_like(cue_arr)
        spikes = np.array(
            [neuron.step(float(feedback[index])) for index, neuron in enumerate(neurons)],
            dtype=np.float64,
        )
        local_trace = local_delay_decay * local_trace + local_spike_gain_delay * spikes
        if use_shared_state:
            assert write_matrix is not None
            shared_state = shared_delay_decay * shared_state + shared_write_gain_delay * (
                write_matrix @ spikes
            )

    recall_spikes: npt.NDArray[np.float64] = np.zeros(cue_arr.size, dtype=np.float64)
    for _ in range(recall_steps):
        if use_shared_state:
            assert read_matrix is not None
            feedback = (read_matrix @ shared_state) * shared_recall_gain
        else:
            feedback = np.zeros_like(cue_arr)
        spikes = np.array(
            [
                neuron.step(float(local_trace_gain_recall * local_trace[index] + feedback[index]))
                for index, neuron in enumerate(neurons)
            ],
            dtype=np.float64,
        )
        recall_spikes += spikes
        local_trace = local_recall_decay * local_trace + local_spike_gain_recall * spikes
        if use_shared_state:
            assert write_matrix is not None
            shared_state = shared_recall_decay * shared_state + shared_write_gain_recall * (
                write_matrix @ spikes
            )

    recalled = (recall_spikes >= 1.0).astype(np.float64)
    accuracy = float(np.mean(recalled == cue_arr))
    return accuracy, recalled


def _run_delayed_recall_suite(
    delay_steps: int,
    *,
    shared_state_dim: int = 3,
    cues: np.ndarray[Any, Any] | None = None,
    seed_count: int = 12,
) -> dict[str, Any]:
    delay_steps = _validated_delay_steps(delay_steps)
    shared_state_dim = _validated_positive_count("shared_state_dim", shared_state_dim)
    seed_count = _validated_positive_count("seed_count", seed_count)
    cue_matrix = _validated_cue_matrix(cues)
    n_neurons = cue_matrix.shape[1]
    accuracies: list[float] = []
    per_cue_accuracies = np.zeros(cue_matrix.shape[0], dtype=np.float64)
    first_recalled: np.ndarray[Any, Any] | None = None

    for seed in range(seed_count):
        rng = np.random.default_rng(seed)
        write_matrix = rng.normal(scale=0.35, size=(shared_state_dim, n_neurons))
        read_matrix = write_matrix.T
        for cue_index, cue in enumerate(cue_matrix):
            accuracy, recalled = _run_delayed_recall_trial(
                cue,
                delay_steps=delay_steps,
                write_matrix=write_matrix,
                read_matrix=read_matrix,
                seed=seed,
            )
            accuracies.append(accuracy)
            per_cue_accuracies[cue_index] += accuracy
            if seed == 0 and cue_index == 0:
                first_recalled = recalled

    per_cue_accuracies /= seed_count
    return {
        "delay_steps": int(delay_steps),
        "cue_count": int(cue_matrix.shape[0]),
        "seed_count": int(seed_count),
        "shared_state_dim": int(shared_state_dim),
        "mean_accuracy": float(np.mean(accuracies)),
        "std_accuracy": float(np.std(accuracies)),
        "per_cue_accuracy": per_cue_accuracies.tolist(),
        "first_cue_target": cue_matrix[0].astype(int).tolist(),
        "first_cue_recalled": (
            cue_matrix[0].astype(int).tolist()
            if first_recalled is None
            else first_recalled.astype(int).tolist()
        ),
    }


def _delayed_recall_local_baseline(
    delay_steps: int,
    *,
    cues: np.ndarray[Any, Any] | None = None,
    seed_count: int = 12,
    shared_state_dim: int = 3,
) -> dict[str, Any]:
    del shared_state_dim
    delay_steps = _validated_delay_steps(delay_steps)
    seed_count = _validated_positive_count("seed_count", seed_count)
    cue_matrix = _validated_cue_matrix(cues)
    accuracies: list[float] = []
    per_cue_accuracies = np.zeros(cue_matrix.shape[0], dtype=np.float64)
    first_recalled: np.ndarray[Any, Any] | None = None

    for seed in range(seed_count):
        for cue_index, cue in enumerate(cue_matrix):
            accuracy, recalled = _run_delayed_recall_trial(
                cue,
                delay_steps=delay_steps,
                write_matrix=None,
                read_matrix=None,
                seed=seed,
            )
            accuracies.append(accuracy)
            per_cue_accuracies[cue_index] += accuracy
            if seed == 0 and cue_index == 0:
                first_recalled = recalled

    per_cue_accuracies /= seed_count
    return {
        "delay_steps": int(delay_steps),
        "cue_count": int(cue_matrix.shape[0]),
        "seed_count": int(seed_count),
        "shared_state_dim": 0,
        "mean_accuracy": float(np.mean(accuracies)),
        "std_accuracy": float(np.std(accuracies)),
        "per_cue_accuracy": per_cue_accuracies.tolist(),
        "first_cue_target": cue_matrix[0].astype(int).tolist(),
        "first_cue_recalled": (
            cue_matrix[0].astype(int).tolist()
            if first_recalled is None
            else first_recalled.astype(int).tolist()
        ),
    }


def _delayed_recall_shared_state_candidate(
    delay_steps: int,
    *,
    cues: np.ndarray[Any, Any] | None = None,
    seed_count: int = 12,
    shared_state_dim: int = 3,
) -> dict[str, Any]:
    return _run_delayed_recall_suite(
        delay_steps,
        cues=cues,
        seed_count=seed_count,
        shared_state_dim=shared_state_dim,
    )


def _compare_delayed_recall_gain(
    baseline: Any,
    candidate: Any,
    config: AlternativePathConfig,
) -> ComparisonStats:
    baseline_delay = int(baseline["delay_steps"])
    candidate_delay = int(candidate["delay_steps"])
    if baseline_delay != candidate_delay:
        return ComparisonStats(
            matched=False,
            comparable_leaf_count=0,
            max_abs_diff=None,
            max_rel_diff=None,
            detail="memory.delayed-recall: delay mismatch between baseline and candidate",
        )

    baseline_accuracy = float(baseline["mean_accuracy"])
    candidate_accuracy = float(candidate["mean_accuracy"])
    improvement = candidate_accuracy - baseline_accuracy
    required_gain = 0.1 if baseline_delay >= 8 else 0.0
    matched = improvement + config.absolute_tolerance >= required_gain
    rel_gain = improvement / max(abs(baseline_accuracy), config.absolute_tolerance)
    detail = (
        "memory.delayed-recall: shared-state candidate improved recall"
        if matched
        else "memory.delayed-recall: shared-state candidate failed to improve recall enough"
    )
    return ComparisonStats(
        matched=matched,
        comparable_leaf_count=2,
        max_abs_diff=abs(improvement),
        max_rel_diff=abs(rel_gain),
        detail=detail,
    )


def make_delayed_recall_shared_state_route() -> AlternativePathRoute[dict[str, Any]]:
    """Route a bounded delayed-recall task against a shared-memory candidate.

    The candidate is *quantum-inspired* only in the broad architectural sense:
    it adds a non-local shared latent state to a real spiking baseline. It does
    not claim to model ATP, Posner molecules, or validated in-vivo quantum
    memory.
    """
    return AlternativePathRoute(
        name="memory.delayed-recall.shared-state",
        baseline=_delayed_recall_local_baseline,
        candidate=_delayed_recall_shared_state_candidate,
        summary=(
            "Local trace-only delayed recall vs a quantum-inspired non-local "
            "shared-state memory candidate"
        ),
        expected_behavior=(
            "The shared-state candidate should equal or exceed the local baseline "
            "on delayed cue recall, with a material gain on longer delays"
        ),
        comparator=_compare_delayed_recall_gain,
    )

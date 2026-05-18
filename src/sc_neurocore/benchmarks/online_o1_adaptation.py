# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Online O(1) adaptation benchmark

"""Deterministic adaptation benchmark for bounded Online O(1) learning."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol

from sc_neurocore._native.learning_bridge import RustOnlineO1Synapse, is_available
from sc_neurocore.hdl_gen.online_learning_emitter import OnlineO1LearningEmitter
from sc_neurocore.learning.online_o1 import OnlineO1Config, OnlineO1Synapse

ONLINE_O1_ADAPTATION_BENCHMARK_SCHEMA_VERSION = "sc-neurocore.online-o1-adaptation-benchmark.v1"


class _OnlineO1Runner(Protocol):
    def step(self, *, pre_spike: bool, post_spike: bool, reward: int) -> Any: ...


def build_online_o1_adaptation_benchmark(
    *,
    config: OnlineO1Config | None = None,
    n_synapses: int = 1024,
    target_weight: int = 192,
    max_pairings: int = 16,
    reward: int = 7,
) -> dict[str, Any]:
    """Return a deterministic Python/Rust adaptation benchmark report."""

    cfg = (
        config if config is not None else OnlineO1Config(weight_bits=8, trace_bits=6, reward_bits=4)
    )
    if n_synapses <= 0:
        raise ValueError("n_synapses must be a positive integer")
    if target_weight < 0 or target_weight > cfg.max_weight:
        raise ValueError("target_weight must fit the configured weight range")
    if max_pairings <= 0:
        raise ValueError("max_pairings must be a positive integer")

    python_result = _run_pairing_protocol(
        OnlineO1Synapse(config=cfg, initial_weight=0),
        target_weight=target_weight,
        max_pairings=max_pairings,
        reward=reward,
    )
    rust_result = _rust_pairing_protocol(
        cfg,
        target_weight=target_weight,
        max_pairings=max_pairings,
        reward=reward,
    )
    return {
        "schema_version": ONLINE_O1_ADAPTATION_BENCHMARK_SCHEMA_VERSION,
        "evidence_class": "deterministic_simulation",
        "hardware_measurement_claimed": False,
        "protocol": "pre_post_reward_pairing",
        "target_weight": target_weight,
        "max_pairings": max_pairings,
        "reward": reward,
        "config": cfg.to_scnir_annotation(rule_id="online_o1_adaptation_benchmark"),
        "resource_estimate": OnlineO1LearningEmitter(config=cfg)
        .estimate_resources(n_synapses=n_synapses, target="artix7")
        .as_dict(),
        "python": python_result,
        "rust": rust_result,
        "parity": {
            "rust_matches_python": (
                bool(rust_result["available"])
                and rust_result["weight_trace"] == python_result["weight_trace"]
                and rust_result["steps_to_target"] == python_result["steps_to_target"]
            )
        },
    }


def write_online_o1_adaptation_benchmark(
    path: str | Path,
    *,
    config: OnlineO1Config | None = None,
    n_synapses: int = 1024,
    target_weight: int = 192,
    max_pairings: int = 16,
    reward: int = 7,
) -> Path:
    """Write a canonical benchmark report and return the output path."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = build_online_o1_adaptation_benchmark(
        config=config,
        n_synapses=n_synapses,
        target_weight=target_weight,
        max_pairings=max_pairings,
        reward=reward,
    )
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def _rust_pairing_protocol(
    config: OnlineO1Config,
    *,
    target_weight: int,
    max_pairings: int,
    reward: int,
) -> dict[str, Any]:
    if not is_available():
        return {
            "available": False,
            "steps_to_target": None,
            "final_weight": None,
            "weight_trace": [],
            "reason": "native Rust learning library is not loaded",
        }
    try:
        runner = RustOnlineO1Synapse(
            weight_bits=config.weight_bits,
            trace_bits=config.trace_bits,
            reward_bits=config.reward_bits,
            learning_shift=config.learning_shift,
            trace_decay_shift=config.trace_decay_shift,
            initial_weight=0,
        )
    except RuntimeError as exc:
        return {
            "available": False,
            "steps_to_target": None,
            "final_weight": None,
            "weight_trace": [],
            "reason": str(exc),
        }
    result = _run_pairing_protocol(
        runner,
        target_weight=target_weight,
        max_pairings=max_pairings,
        reward=reward,
    )
    result["available"] = True
    return result


def _run_pairing_protocol(
    runner: _OnlineO1Runner,
    *,
    target_weight: int,
    max_pairings: int,
    reward: int,
) -> dict[str, Any]:
    weight_trace: list[int] = []
    steps_to_target: int | None = None
    final_weight = 0
    for pair_index in range(max_pairings):
        runner.step(pre_spike=True, post_spike=False, reward=0)
        snapshot = runner.step(pre_spike=False, post_spike=True, reward=reward)
        final_weight = int(snapshot.weight)
        weight_trace.append(final_weight)
        if final_weight >= target_weight and steps_to_target is None:
            steps_to_target = (pair_index + 1) * 2
            break
    return {
        "available": True,
        "steps_to_target": steps_to_target,
        "final_weight": final_weight,
        "weight_trace": weight_trace,
    }

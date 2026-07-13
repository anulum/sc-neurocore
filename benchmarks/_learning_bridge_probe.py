#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Isolated autonomous-learning benchmark probe

"""Emit one deterministic Python, Rust, Torch, Go, and Julia timing sample."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable, TypeVar

import numpy as np

T = TypeVar("T")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=4096)
    return parser


def _events(steps: int) -> tuple[np.ndarray[Any, Any], ...]:
    indices = np.arange(steps)
    pre = np.ascontiguousarray(indices % 17 == 0, dtype=np.bool_)
    post = np.ascontiguousarray(indices % 23 == 3, dtype=np.bool_)
    rewards = np.ascontiguousarray(((indices % 11) - 5) / 10.0, dtype=np.float32)
    return pre, post, rewards


def _timed(operation: Callable[[], T]) -> tuple[int, T]:
    started = time.perf_counter_ns()
    result = operation()
    return time.perf_counter_ns() - started, result


def _go_weight(root: Path, steps: int) -> tuple[int | None, float | None]:
    if shutil.which("go") is None:
        return None, None
    bridge_source = (
        root / "src/sc_neurocore/accel/go/autonomous_learning/learning_bridge.go"
    ).read_text(encoding="utf-8")
    signature = next(
        line for line in bridge_source.splitlines() if "PlasticityRule) StepDt" in line
    )
    call = "rule.StepDt(i % 17 == 0, i % 23 == 3, float32((i % 11) - 5) / 10.0, 0.001)"
    if " error {" in signature:
        step = f"""if err := {call}; err != nil {{
            log.Fatal(err)
        }}"""
    else:
        step = call
    source = f"""package main
import (
    "fmt"
    "log"
    learning "github.com/anulum/sc-neurocore/accel/autonomous_learning"
)
func main() {{
    rule := learning.NewPlasticityRule(learning.RuleStdp, 0.5, 0.01, 20.0)
    if rule == nil {{ log.Fatal("nil learning rule") }}
    defer rule.Destroy()
    for i := 0; i < {steps}; i++ {{
        {step}
    }}
    fmt.Printf("%.9f\\n", rule.Weight())
}}
"""
    with tempfile.TemporaryDirectory(prefix="sc-learning-go-") as directory:
        script = Path(directory) / "main.go"
        script.write_text(source, encoding="utf-8")
        environment = os.environ.copy()
        environment["CGO_ENABLED"] = "1"
        library = Path(environment["SC_NEUROCORE_LIB_PATH"]).resolve()
        environment["CGO_LDFLAGS"] = f"-L{library.parent}"
        environment["LD_LIBRARY_PATH"] = str(library.parent)
        started = time.perf_counter_ns()
        completed = subprocess.run(
            ["go", "run", str(script)],
            cwd=root / "src/sc_neurocore/accel/go",
            env=environment,
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
        )
        elapsed = time.perf_counter_ns() - started
    if completed.returncode != 0:
        raise RuntimeError(f"Go probe failed: {completed.stderr.strip()}")
    return elapsed, float(completed.stdout.strip().splitlines()[-1])


def _julia_weight(root: Path, steps: int) -> tuple[int | None, float | None]:
    if shutil.which("julia") is None:
        return None, None
    bridge = json.dumps(str(root / "src/sc_neurocore/accel/julia/_native/learning_bridge.jl"))
    source = f"""
include({bridge})
using .LearningBridgeAccel
rule = LearningBridgeAccel.RustPlasticityRule(
    LearningBridgeAccel.RULE_STDP, 0.5f0, 0.01f0, 20.0f0
)
for i in 0:{steps - 1}
    LearningBridgeAccel.step(
        rule, i % 17 == 0, i % 23 == 3, Float32((i % 11) - 5) / 10.0f0, 0.001f0
    )
end
println(LearningBridgeAccel.weight(rule))
LearningBridgeAccel.destroy_rule(rule)
"""
    started = time.perf_counter_ns()
    completed = subprocess.run(
        ["julia", "--startup-file=no", "-e", source],
        cwd=root,
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    elapsed = time.perf_counter_ns() - started
    if completed.returncode != 0:
        raise RuntimeError(f"Julia probe failed: {completed.stderr.strip()}")
    return elapsed, float(completed.stdout.strip().splitlines()[-1])


def _python_paths(steps: int) -> tuple[dict[str, int], dict[str, Any]]:
    import_started = time.perf_counter_ns()
    from sc_neurocore._native.learning_bridge import (
        RULE_STDP,
        RustPlasticityRule,
        RustRuleLayer,
        TorchRuleLayer,
        is_available,
    )

    import_ns = time.perf_counter_ns() - import_started
    if not is_available():
        raise RuntimeError("Python could not load the requested autonomous-learning library")
    pre, post, rewards = _events(steps)

    scalar = RustPlasticityRule(RULE_STDP, 0.5, 0.01, 20.0)

    def scalar_work() -> float:
        for index in range(steps):
            scalar.step(bool(pre[index]), bool(post[index]), 0.001, float(rewards[index]))
        return scalar.weight

    scalar_ns, scalar_weight = _timed(scalar_work)
    batched = RustPlasticityRule(RULE_STDP, 0.5, 0.01, 20.0)
    batched_ns, _ = _timed(lambda: batched.step_batched(pre, post, rewards, 0.001))
    layer = RustRuleLayer(steps, RULE_STDP, 0.5, 0.01, 20.0)
    layer_ns, _ = _timed(lambda: layer.step(pre, post, rewards, 0.001))
    layer_weights = layer.get_weights()
    layer_state = bytes(layer.get_state_dict()["mem_buffer"])

    import torch

    torch_layer = TorchRuleLayer(
        1,
        RULE_STDP,
        0.5,
        0.01,
        20.0,
        False,
        param_a_minus=0.005,
        tau_plus=20.0,
        tau_minus=20.0,
    )
    pre_t = torch.from_numpy(pre.astype(np.float32)).reshape(steps, 1)
    post_t = torch.from_numpy(post.astype(np.float32)).reshape(steps, 1)
    rewards_t = torch.from_numpy(rewards).reshape(steps, 1)

    def torch_work() -> float:
        for index in range(steps):
            torch_layer.forward(pre_t[index], post_t[index], rewards_t[index], 0.001)
        return float(torch_layer.weights[0])

    torch_ns, torch_weight = _timed(torch_work)
    outputs = {
        "rust_scalar_weight": scalar_weight,
        "rust_batched_weight": batched.weight,
        "torch_weight": torch_weight,
        "layer_weights_sha256": hashlib.sha256(layer_weights.tobytes()).hexdigest(),
        "layer_state_sha256": hashlib.sha256(layer_state).hexdigest(),
    }
    timings = {
        "import_ns": import_ns,
        "rust_scalar_ns": scalar_ns,
        "rust_batched_ns": batched_ns,
        "rust_layer_ns": layer_ns,
        "torch_ns": torch_ns,
    }
    return timings, outputs


def main() -> int:
    args = _parser().parse_args()
    if args.steps <= 0:
        raise ValueError("steps must be positive")
    timings, outputs = _python_paths(args.steps)
    go_ns, go_weight = _go_weight(args.root, args.steps)
    julia_ns, julia_weight = _julia_weight(args.root, args.steps)
    timings.update({"go_process_ns": go_ns, "julia_process_ns": julia_ns})
    outputs.update({"go_weight": go_weight, "julia_weight": julia_weight})
    reference = float(outputs["rust_scalar_weight"])
    for name in ("rust_batched_weight", "torch_weight", "go_weight", "julia_weight"):
        value = outputs[name]
        if value is not None and not np.isclose(reference, float(value), rtol=1e-5, atol=1e-6):
            raise RuntimeError(f"cross-language weight mismatch for {name}: {value} vs {reference}")
    canonical_outputs = {
        name: round(float(value), 7) if name.endswith("_weight") and value is not None else value
        for name, value in outputs.items()
    }
    canonical = json.dumps(
        canonical_outputs, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    rss_kib = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        rss_kib //= 1024
    print(
        json.dumps(
            {
                "timings": timings,
                "outputs": outputs,
                "canonical_sha256": hashlib.sha256(canonical).hexdigest(),
                "canonical_bytes": len(canonical),
                "max_rss_kib": rss_kib,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

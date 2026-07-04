# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Zenith GPU end-to-end benchmark

import os
import sys
import time
import json
import argparse
import psutil
import tempfile
import numpy as np
import torch
import torch.utils.benchmark as benchmark
from typing import Dict, Any

from sc_neurocore.plasticity import create_plasticity_layer
from sc_neurocore._native.learning_bridge import (
    RULE_ELIGENT,
    RULE_STDP,
    RULE_REWARD_STDP,
    RULE_BCM,
    set_deterministic_mode,
)


def benchmark_torch_autograd(scale: int, rule: int, device: str) -> Dict[str, Any]:
    print(
        f"  [Torch] Benchmarking Surrogate Autograd (Rule: {rule}, Scale: {scale}, Device: {device})"
    )

    # Pre-allocate
    dt = 1.0
    pre_spikes = torch.randint(0, 2, (scale,), dtype=torch.bool, device=device)
    post_spikes = torch.randint(0, 2, (scale,), dtype=torch.bool, device=device)
    rewards = torch.rand((scale,), dtype=torch.float32, device=device)

    layer = create_plasticity_layer(
        count=scale, rule_type=rule, backend="torch", autograd=True, param_a=0.1, param_b=0.05
    )
    if device == "cuda":
        layer.cuda()

    # Warmup
    out = layer(pre_spikes, post_spikes, rewards, dt)
    loss = out.sum()
    loss.backward()
    if device == "cuda":
        torch.cuda.synchronize()

    layer.pre_trace.zero_()
    layer.post_trace.zero_()

    # Measure Forward
    t0_fwd = benchmark.Timer(
        stmt="layer(pre_spikes, post_spikes, rewards, dt)",
        globals={
            "layer": layer,
            "pre_spikes": pre_spikes,
            "post_spikes": post_spikes,
            "rewards": rewards,
            "dt": dt,
        },
    )
    res_fwd = t0_fwd.timeit(50)

    # Measure Backward
    # To measure backward, we need to run forward, then backward, then subtract forward.
    # Or just measure backward via timeit inside a loop
    def fwd_bwd():
        out = layer(pre_spikes, post_spikes, rewards, dt)
        out.sum().backward()

    t0_bwd = benchmark.Timer(stmt="fwd_bwd()", globals={"fwd_bwd": fwd_bwd})
    res_bwd = t0_bwd.timeit(50)

    fwd_latency_ms = res_fwd.mean * 1000.0
    total_latency_ms = res_bwd.mean * 1000.0
    bwd_latency_ms = total_latency_ms - fwd_latency_ms

    return {
        "forward_ms": fwd_latency_ms,
        "backward_ms": bwd_latency_ms,
        "total_ms": total_latency_ms,
    }


def benchmark_rust_physics(
    scale: int, rule: int, force_deterministic: bool = True, backend="rust"
) -> Dict[str, Any]:
    print(
        f"  [{backend.upper()}] Benchmarking Native Physics (Rule: {rule}, Scale: {scale}, Deterministic: {force_deterministic})"
    )

    if force_deterministic:
        set_deterministic_mode(seed=42)

    pre_spikes = (np.random.rand(scale) < 0.1).tolist()
    post_spikes = (np.random.rand(scale) < 0.1).tolist()
    rewards = (np.random.rand(scale) * 0.1).tolist()
    dt = 1.0

    layer = create_plasticity_layer(
        count=scale, rule_type=rule, backend=backend, param_a=0.1, param_b=0.05
    )

    # Warmup
    layer.step(pre_spikes, post_spikes, rewards, dt)

    iters = 10
    start_t = time.perf_counter()
    for _ in range(iters):
        layer.step(pre_spikes, post_spikes, rewards, dt)
    end_t = time.perf_counter()

    avg_latency_ms = ((end_t - start_t) * 1000.0) / iters
    return {"step_ms": avg_latency_ms, "deterministic": force_deterministic}


def benchmark_exascale_io(scale: int, rule: int) -> Dict[str, Any]:
    print(f"  [IO] Benchmarking Exascale Persistence (Rule: {rule}, Scale: {scale})")

    layer = create_plasticity_layer(count=scale, rule_type=rule, backend="rust")

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        save_path = tmp.name

    try:
        # Write Benchmark
        start_w = time.perf_counter()
        layer.save(save_path)
        end_w = time.perf_counter()
        write_ms = (end_w - start_w) * 1000.0

        file_size_bytes = os.path.getsize(save_path)
        file_size_mb = file_size_bytes / (1024 * 1024)
        write_mb_s = file_size_mb / ((end_w - start_w) or 1e-9)

        # Read Benchmark
        new_layer = create_plasticity_layer(count=scale, rule_type=rule, backend="rust")
        start_r = time.perf_counter()
        new_layer.load(save_path)
        end_r = time.perf_counter()
        read_ms = (end_r - start_r) * 1000.0
        read_mb_s = file_size_mb / ((end_r - start_r) or 1e-9)

    finally:
        os.unlink(save_path)

    return {
        "write_ms": write_ms,
        "write_mb_s": write_mb_s,
        "read_ms": read_ms,
        "read_mb_s": read_mb_s,
        "file_size_mb": file_size_mb,
    }


def check_parity(scale: int) -> bool:
    print(f"  [Parity] Executing formal deterministic properties test (Scale: {scale})")
    set_deterministic_mode(seed=42)

    pre_spikes = (np.random.rand(scale) < 0.1).tolist()
    post_spikes = (np.random.rand(scale) < 0.1).tolist()
    rewards = (np.random.rand(scale) * 0.1).tolist()
    dt = 1.0

    torch_layer = create_plasticity_layer(
        count=scale, rule_type=RULE_STDP, backend="torch", weight=0.5
    )
    rust_layer = create_plasticity_layer(
        count=scale, rule_type=RULE_STDP, backend="rust", weight=0.5
    )

    # Step Torch
    t_pre = torch.tensor(pre_spikes, dtype=torch.bool)
    t_post = torch.tensor(post_spikes, dtype=torch.bool)
    t_rew = torch.tensor(rewards, dtype=torch.float32)
    torch_layer(t_pre, t_post, t_rew, dt)
    torch_weights_1 = torch_layer.get_weights()

    # Step Rust
    rust_layer.step(pre_spikes, post_spikes, rewards, dt)
    rust_weights_1 = rust_layer.get_weights()

    # Compare
    is_close = np.allclose(torch_weights_1, rust_weights_1, atol=1e-6)

    # IO Parity check
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        save_path = tmp.name

    try:
        rust_layer.save(save_path)
        rust_load_layer = create_plasticity_layer(count=scale, rule_type=RULE_STDP, backend="rust")
        rust_load_layer.load(save_path)
        rust_load_weights = rust_load_layer.get_weights()
        is_load_close = np.allclose(rust_weights_1, rust_load_weights, atol=1e-6)
    finally:
        os.unlink(save_path)

    print(f"    - Torch/Rust step parity: {'PASS' if is_close else 'FAIL'}")
    print(f"    - Rust IO persistence parity: {'PASS' if is_load_close else 'FAIL'}")
    return is_close and is_load_close


def main():
    parser = argparse.ArgumentParser(description="Zenith End-to-End Benchmarks")
    parser.add_argument(
        "--scale",
        type=int,
        nargs="+",
        default=[1_000_000],
        help="Scales to test (e.g. 100000 1000000)",
    )
    parser.add_argument(
        "--rules", type=str, default="all", help="Rules to test: 'all' or 'stdp,bcm'"
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="all",
        help="Native backend to test against torch ('rust', 'rust-wgpu', or 'all')",
    )
    parser.add_argument(
        "--output", type=str, default="zenith_benchmarks.json", help="JSON output file."
    )
    parser.add_argument("--ci", action="store_true", help="Run in CI mode (errors out if failure).")
    parser.add_argument(
        "--quick", action="store_true", help="Quick mode (100k nodes overrides scale)."
    )
    args = parser.parse_args()

    scales = [100_000] if args.quick else args.scale

    rule_map = {
        "eligent": RULE_ELIGENT,
        "stdp": RULE_STDP,
        "r_stdp": RULE_REWARD_STDP,
        "bcm": RULE_BCM,
    }

    if args.rules == "all":
        rules = list(rule_map.values())
        rule_names = list(rule_map.keys())
    else:
        rule_names = args.rules.split(",")
        rules = [rule_map[r] for r in rule_names]

    if not torch.cuda.is_available():
        print(
            "CUDA ERROR: The explicit GPU benchmark `bench_zenith_e2e_gpu.py` requires an active CUDA GPU environment."
        )
        print("Please run `bench_zenith_e2e.py` instead for standard CPU fallbacks.")
        sys.exit(1)

    device = "cuda"
    results = {}

    # Always set deterministic at start
    set_deterministic_mode(seed=42)

    for scale in scales:
        print(f"\n=== SCALE: {scale} nodes ===")
        results[scale] = {}

        # 1. PARITY
        parity_pass = check_parity(scale)
        if args.ci and not parity_pass:
            print("CI Error: Parity checks failed!")
            sys.exit(1)

        results[scale]["parity"] = parity_pass

        # 2. RUN SUITE
        for rule_name, rule_id in zip(rule_names, rules):
            print(f"\n--- Outputting Bounds for Rule: {rule_name.upper()} ---")

            # 2a. Torch Autograd
            torch_res = benchmark_torch_autograd(scale, rule_id, device)

            # 2b. Native Physics (Deterministic)
            results[scale][rule_name] = {"torch": torch_res}
            if args.backend in ["rust", "all"]:
                rust_res_det = benchmark_rust_physics(
                    scale, rule_id, force_deterministic=True, backend="rust"
                )
                results[scale][rule_name]["rust"] = rust_res_det
            if args.backend in ["rust-wgpu", "all"]:
                rust_wgpu_res = benchmark_rust_physics(
                    scale, rule_id, force_deterministic=True, backend="rust-wgpu"
                )
                results[scale][rule_name]["rust-wgpu"] = rust_wgpu_res

            # 2c. Exascale IO
            io_res = benchmark_exascale_io(scale, rule_id)
            results[scale][rule_name]["io"] = io_res

    # Output JSON
    with open(args.output, "w") as f:
        json.dump(
            {
                "timestamp": time.time(),
                "device": device,
                "system_ram_gb": psutil.virtual_memory().total / (1024**3),
                "cpu_cores": psutil.cpu_count(logical=True),
                "results": results,
            },
            f,
            indent=2,
        )

    print(f"\n[OK] Benchmarks complete. Data dumped to {args.output}")


if __name__ == "__main__":
    main()

# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Source/config provenance header

import time
import numpy as np

try:
    from sc_neurocore._native.learning_bridge import (
        is_available,
        RustPlasticityRule,
        RustRuleLayer,
        TorchRuleLayer,
        RULE_STDP,
    )

    FFI_AVAILABLE = is_available()
except ImportError:
    FFI_AVAILABLE = False


def run_benchmark(timesteps: int = 100_000):
    print("Benchmarking Autonomous Learning (Rust FFI over Python via C-Types)...")
    print(f"Timesteps: {timesteps}")

    if not FFI_AVAILABLE:
        print("Rust FFI is not available. Skipping benchmark.")
        return

    # Create deterministic synthetic spike trains (10% firing rate uniformly distributed)
    rng = np.random.default_rng(42)
    pre_spikes = rng.random(timesteps) < 0.1
    post_spikes = rng.random(timesteps) < 0.1
    rewards = rng.random(timesteps) * 0.1

    # Sequential Execution
    rule_seq = RustPlasticityRule(rule_type=RULE_STDP, weight=0.5, param_a=0.1, param_b=0.05)
    start_t_seq = time.perf_counter()
    for t in range(timesteps):
        rule_seq.step(bool(pre_spikes[t]), bool(post_spikes[t]), float(rewards[t]))
    end_t_seq = time.perf_counter()

    elapsed_seq_ms = (end_t_seq - start_t_seq) * 1000.0
    ns_per_step_seq = (elapsed_seq_ms * 1e6) / timesteps

    # Batched Execution
    rule_batched = RustPlasticityRule(rule_type=RULE_STDP, weight=0.5, param_a=0.1, param_b=0.05)
    start_t_batched = time.perf_counter()
    rule_batched.step_batched(pre_spikes, post_spikes, rewards)
    end_t_batched = time.perf_counter()

    elapsed_batched_ms = (end_t_batched - start_t_batched) * 1000.0
    ns_per_step_batched = (elapsed_batched_ms * 1e6) / timesteps

    # Spatial Layer Parallelization Execution
    layer_spatial = RustRuleLayer(
        count=timesteps, rule_type=RULE_STDP, weight=0.5, param_a=0.1, param_b=0.05
    )
    start_t_spatial = time.perf_counter()
    layer_spatial.step(pre_spikes, post_spikes, rewards)
    end_t_spatial = time.perf_counter()

    elapsed_spatial_ms = (end_t_spatial - start_t_spatial) * 1000.0
    ns_per_step_spatial = (elapsed_spatial_ms * 1e6) / timesteps

    # Check that spatial mapping works correctly, index 0 weight should match batched.
    spatial_weights = layer_spatial.get_weights()
    spatial_w0 = spatial_weights[0] if len(spatial_weights) > 0 else 0.0

    # GPU Tensor Parallelization Execution (CUDA/ROCm)
    try:
        import torch

        has_torch = True
    except ImportError:
        has_torch = False

    if has_torch:
        layer_gpu = TorchRuleLayer(
            count=timesteps, rule_type=RULE_STDP, weight=0.5, param_a=0.1, param_b=0.05
        )
        # Pre-cache tensors onto GPU so data-transfer time is not counted in kernel dispatch execution time
        gpu_pre = torch.tensor(pre_spikes, device=layer_gpu._device, dtype=torch.bool)
        gpu_post = torch.tensor(post_spikes, device=layer_gpu._device, dtype=torch.bool)
        gpu_rew = torch.tensor(rewards, device=layer_gpu._device, dtype=torch.float32)

        # Warm-up (important for CUDA to initialize contexts)
        layer_gpu.step(gpu_pre, gpu_post, gpu_rew)
        # Reset trace for accurate bench
        layer_gpu._pre_trace.zero_()
        layer_gpu._post_trace.zero_()
        layer_gpu._weights.fill_(0.5)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        start_t_gpu = time.perf_counter()
        layer_gpu.step(gpu_pre, gpu_post, gpu_rew)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end_t_gpu = time.perf_counter()

        elapsed_gpu_ms = (end_t_gpu - start_t_gpu) * 1000.0
        ns_per_step_gpu = (elapsed_gpu_ms * 1e6) / timesteps
        gpu_w0 = layer_gpu.get_weights()[0]
    else:
        elapsed_gpu_ms = 0.0
        ns_per_step_gpu = 0.0
        gpu_w0 = 0.0

    print("-" * 50)
    print("SEQUENTIAL EXECUTION")
    print(f"Total time   : {elapsed_seq_ms:.4f} ms")
    print(f"Latency/step : {ns_per_step_seq:.4f} ns")
    print(f"Final weight : {rule_seq.weight:.4f}")
    print("-" * 50)
    print("BATCHED TEMPORAL VECTORIZATION (Single Thread Loop)")
    print(f"Total time   : {elapsed_batched_ms:.4f} ms")
    print(f"Latency/step : {ns_per_step_batched:.4f} ns")
    print(f"Final weight : {rule_batched.weight:.4f}")
    if elapsed_batched_ms > 0:
        speedup = elapsed_seq_ms / elapsed_batched_ms
        print(f"-> Speedup Factor: {speedup:.2f}X over Sequential")
    print("-" * 50)
    print("SPATIAL LAYER PARALLELIZATION (Rayon Multi-Core)")
    print(f"Total time   : {elapsed_spatial_ms:.4f} ms")
    print(f"Latency/step : {ns_per_step_spatial:.4f} ns")
    print(f"Sample weight: {spatial_w0:.4f}")
    if elapsed_spatial_ms > 0:
        speedup = elapsed_seq_ms / elapsed_spatial_ms
        print(f"-> Speedup Factor: {speedup:.2f}X over Sequential")
        speedup2 = elapsed_batched_ms / elapsed_spatial_ms
        print(f"-> Multicore Scale: {speedup2:.2f}X over Batched Single Thread")
    print("-" * 50)

    if has_torch:
        device_name = "CUDA" if torch.cuda.is_available() else "CPU"
        print(f"GPU TENSOR EXECUTION (PyTorch {device_name})")
        print(f"Total time   : {elapsed_gpu_ms:.4f} ms")
        print(f"Latency/step : {ns_per_step_gpu:.4f} ns")
        print(f"Sample weight: {gpu_w0:.4f}")
        if elapsed_gpu_ms > 0:
            speedup = elapsed_seq_ms / elapsed_gpu_ms
            print(f"-> Speedup Factor: {speedup:.2f}X over Sequential")
            speedup2 = elapsed_batched_ms / elapsed_gpu_ms
            print(f"-> GPU Bound Scale: {speedup2:.2f}X over Batched Single Thread")
            speedup3 = elapsed_spatial_ms / elapsed_gpu_ms
            print(f"-> GPU Bound Scale: {speedup3:.2f}X over Rayon CPU")
        print("-" * 50)


if __name__ == "__main__":
    run_benchmark(10_000_000)

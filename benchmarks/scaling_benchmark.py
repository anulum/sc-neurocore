# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Scaling Benchmark — SC-NeuroCore vs Brian2 vs NEST

"""
Scaling Benchmark — SC-NeuroCore vs Brian2 vs NEST
===================================================

Brunel balanced network (80% exc, 20% inh, Poisson drive) at 4 dynamical
regimes (Brunel 2000, Table 1).  Measures wall-clock, memory, synaptic
throughput, firing rate, and activation sparsity as neuron count scales
from 1K to 100K.

Regimes
-------
SR  synchronous regular    g=3.0  nu_ext/nu_thr=2.0
SI  synchronous irregular  g=6.0  nu_ext/nu_thr=4.0
AI  asynchronous irregular g=5.0  nu_ext/nu_thr=2.0
AR  asynchronous regular   g=3.0  nu_ext/nu_thr=5.0

Usage::

    python benchmarks/scaling_benchmark.py                        # defaults
    python benchmarks/scaling_benchmark.py --scales 1000 5000
    python benchmarks/scaling_benchmark.py --regimes AI AR
    python benchmarks/scaling_benchmark.py --json results.json --markdown
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import time
import tracemalloc
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class RunMetrics:
    wall_time_s: float
    peak_rss_mb: float
    total_spikes: int
    mean_rate_hz: float
    synaptic_events: int
    synaptic_events_per_s: float
    activation_sparsity: float
    gpu_mem_mb: float = 0.0


@dataclass
class ScalePoint:
    n_neurons: int
    simulator: str
    regime: str
    n_synapses: int
    runs: list[RunMetrics] = field(default_factory=list)

    @property
    def wall_mean(self) -> float:
        return float(np.mean([r.wall_time_s for r in self.runs]))

    @property
    def wall_std(self) -> float:
        return (
            float(np.std([r.wall_time_s for r in self.runs], ddof=1)) if len(self.runs) > 1 else 0.0
        )

    @property
    def wall_min(self) -> float:
        return float(np.min([r.wall_time_s for r in self.runs]))

    @property
    def peak_rss_mb(self) -> float:
        return float(np.max([r.peak_rss_mb for r in self.runs]))

    @property
    def spikes_mean(self) -> float:
        return float(np.mean([r.total_spikes for r in self.runs]))

    @property
    def rate_mean(self) -> float:
        return float(np.mean([r.mean_rate_hz for r in self.runs]))

    @property
    def syn_events_per_s_mean(self) -> float:
        return float(np.mean([r.synaptic_events_per_s for r in self.runs]))

    @property
    def sparsity_mean(self) -> float:
        return float(np.mean([r.activation_sparsity for r in self.runs]))


# ---------------------------------------------------------------------------
# Brunel regimes — Brunel 2000, Table 1
# ---------------------------------------------------------------------------
BRUNEL_REGIMES = {
    "SR": {"g_inh": 3.0, "eta": 2.0, "label": "synchronous regular"},
    "SI": {"g_inh": 6.0, "eta": 4.0, "label": "synchronous irregular"},
    "AI": {"g_inh": 5.0, "eta": 2.0, "label": "asynchronous irregular"},
    "AR": {"g_inh": 3.0, "eta": 5.0, "label": "asynchronous regular"},
}


@dataclass
class BrunelConfig:
    n_neurons: int
    regime: str = "AI"
    sim_ms: float = 500.0
    dt: float = 0.1
    conn_prob: float = 0.1
    weight_exc: float = 0.1
    v_threshold: float = 20.0
    v_reset: float = 10.0
    v_rest: float = 0.0
    tau_mem: float = 20.0
    seed: int = 42

    @property
    def n_exc(self) -> int:
        return int(self.n_neurons * 0.8)

    @property
    def n_inh(self) -> int:
        return self.n_neurons - self.n_exc

    @property
    def g_inh(self) -> float:
        return BRUNEL_REGIMES[self.regime]["g_inh"]

    @property
    def weight_inh(self) -> float:
        return self.g_inh * self.weight_exc

    @property
    def c_ext(self) -> float:
        """Number of external Poisson connections per neuron (= C_E)."""
        return self.conn_prob * self.n_exc

    @property
    def external_rate_hz(self) -> float:
        """Per-connection external Poisson rate (nu_ext = eta * nu_thr).
        Total external spikes per neuron per second = c_ext * external_rate_hz."""
        ce = self.c_ext
        nu_thr = self.v_threshold / (self.weight_exc * ce * self.tau_mem * 1e-3) if ce > 0 else 20.0
        eta = BRUNEL_REGIMES[self.regime]["eta"]
        return eta * nu_thr

    @property
    def ext_poisson_lambda(self) -> float:
        """Expected external spikes per neuron per timestep = C_E * nu_ext * dt_s."""
        return self.c_ext * self.external_rate_hz * self.dt / 1000.0


# ---------------------------------------------------------------------------
# Memory measurement (cross-platform via tracemalloc)
# ---------------------------------------------------------------------------
def _tracemalloc_peak_mb() -> float:
    _, peak = tracemalloc.get_traced_memory()
    return peak / 1024 / 1024


# ---------------------------------------------------------------------------
# System info
# ---------------------------------------------------------------------------
def _system_info() -> dict:
    info = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu": platform.processor() or "unknown",
        "cpu_count": os.cpu_count(),
    }
    try:
        import torch

        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["cuda_version"] = torch.version.cuda
    except ImportError:
        pass
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    info["cpu"] = line.split(":")[1].strip()
                    break
    except FileNotFoundError:
        pass
    return info


# ---------------------------------------------------------------------------
# Shared Brunel weight-matrix builder
# ---------------------------------------------------------------------------
def _build_weights_dense(cfg: BrunelConfig, rng: np.random.Generator):
    n = cfg.n_neurons
    conn_mask = rng.random((n, n)) < cfg.conn_prob
    np.fill_diagonal(conn_mask, False)
    weights = np.where(conn_mask, cfg.weight_exc, 0.0).astype(np.float32)
    weights[cfg.n_exc :, :] *= -cfg.g_inh
    n_synapses = int(conn_mask.sum())
    return weights, n_synapses


def _compute_extended_metrics(
    spike_count: int,
    n: int,
    sim_ms: float,
    n_synapses: int,
    wall: float,
    step_spike_counts: list[int],
) -> dict:
    rate = spike_count / (sim_ms / 1000.0) / n if n > 0 else 0.0
    # fan_out ≈ n_synapses / n (average post-synaptic targets per neuron)
    fan_out = n_synapses / n if n > 0 else 0.0
    syn_events = int(spike_count * fan_out)
    syn_per_s = syn_events / wall if wall > 0 else 0.0
    # activation sparsity: fraction of neurons silent per step (averaged)
    steps = len(step_spike_counts) if step_spike_counts else 1
    sparsity = (
        float(np.mean([1.0 - sc / n for sc in step_spike_counts])) if step_spike_counts else 1.0
    )
    return {
        "rate": rate,
        "syn_events": syn_events,
        "syn_per_s": syn_per_s,
        "sparsity": sparsity,
    }


# ---------------------------------------------------------------------------
# Simulator: Vectorized NumPy dense
# ---------------------------------------------------------------------------
def run_numpy(cfg: BrunelConfig) -> RunMetrics:
    rng = np.random.default_rng(cfg.seed)
    n = cfg.n_neurons
    weights, n_synapses = _build_weights_dense(cfg, rng)

    v = np.full(n, cfg.v_rest, dtype=np.float32)
    alpha = np.float32(cfg.dt / cfg.tau_mem)
    steps = int(cfg.sim_ms / cfg.dt)
    spike_count = 0
    prev_spikes = np.zeros(n, dtype=bool)
    step_counts: list[int] = []

    gc.collect()
    tracemalloc.start()
    t0 = time.perf_counter()

    for _ in range(steps):
        ext = rng.poisson(cfg.ext_poisson_lambda, n).astype(np.float32)
        I_syn = (
            weights[prev_spikes].sum(axis=0) if prev_spikes.any() else np.zeros(n, dtype=np.float32)
        )
        v += ext * cfg.weight_exc + I_syn
        v += alpha * (cfg.v_rest - v)
        fired = v >= cfg.v_threshold
        sc = int(fired.sum())
        spike_count += sc
        step_counts.append(sc)
        v[fired] = cfg.v_reset
        prev_spikes = fired

    wall = time.perf_counter() - t0
    peak_mb = _tracemalloc_peak_mb()
    tracemalloc.stop()

    m = _compute_extended_metrics(spike_count, n, cfg.sim_ms, n_synapses, wall, step_counts)
    return RunMetrics(
        wall_time_s=wall,
        peak_rss_mb=peak_mb,
        total_spikes=spike_count,
        mean_rate_hz=m["rate"],
        synaptic_events=m["syn_events"],
        synaptic_events_per_s=m["syn_per_s"],
        activation_sparsity=m["sparsity"],
    )


# ---------------------------------------------------------------------------
# Simulator: Sparse NumPy (CSR)
# ---------------------------------------------------------------------------
def run_numpy_sparse(cfg: BrunelConfig) -> RunMetrics:
    from scipy import sparse

    rng = np.random.default_rng(cfg.seed)
    n = cfg.n_neurons
    weights, n_synapses = _build_weights_dense(cfg, rng)
    w_csr = sparse.csr_matrix(weights)
    del weights

    v = np.full(n, cfg.v_rest, dtype=np.float32)
    alpha = np.float32(cfg.dt / cfg.tau_mem)
    steps = int(cfg.sim_ms / cfg.dt)
    spike_count = 0
    fired_indices = np.array([], dtype=np.intp)
    step_counts: list[int] = []

    gc.collect()
    tracemalloc.start()
    t0 = time.perf_counter()

    for _ in range(steps):
        ext = rng.poisson(cfg.ext_poisson_lambda, n).astype(np.float32)
        if fired_indices.size > 0:
            I_syn = np.asarray(w_csr[fired_indices].sum(axis=0)).ravel()
        else:
            I_syn = np.zeros(n, dtype=np.float32)
        v += ext * cfg.weight_exc + I_syn
        v += alpha * (cfg.v_rest - v)
        fired_mask = v >= cfg.v_threshold
        sc = int(fired_mask.sum())
        spike_count += sc
        step_counts.append(sc)
        v[fired_mask] = cfg.v_reset
        fired_indices = np.nonzero(fired_mask)[0]

    wall = time.perf_counter() - t0
    peak_mb = _tracemalloc_peak_mb()
    tracemalloc.stop()

    m = _compute_extended_metrics(spike_count, n, cfg.sim_ms, n_synapses, wall, step_counts)
    return RunMetrics(
        wall_time_s=wall,
        peak_rss_mb=peak_mb,
        total_spikes=spike_count,
        mean_rate_hz=m["rate"],
        synaptic_events=m["syn_events"],
        synaptic_events_per_s=m["syn_per_s"],
        activation_sparsity=m["sparsity"],
    )


# ---------------------------------------------------------------------------
# Simulator: PyTorch CUDA (dense)
# ---------------------------------------------------------------------------
def run_pytorch_cuda(cfg: BrunelConfig) -> RunMetrics | None:
    try:
        import torch

        if not torch.cuda.is_available():
            return None
    except ImportError:
        return None

    device = torch.device("cuda")
    rng = np.random.default_rng(cfg.seed)
    n = cfg.n_neurons
    weights_np, n_synapses = _build_weights_dense(cfg, rng)

    w = torch.tensor(weights_np, dtype=torch.float32, device=device)
    v = torch.full((n,), cfg.v_rest, dtype=torch.float32, device=device)
    alpha = cfg.dt / cfg.tau_mem
    steps = int(cfg.sim_ms / cfg.dt)
    prev_spikes = torch.zeros(n, dtype=torch.float32, device=device)

    # Hoisted constants — avoid per-step GPU tensor creation
    v_reset_t = torch.full((n,), cfg.v_reset, dtype=torch.float32, device=device)
    poisson_rate = torch.tensor(cfg.ext_poisson_lambda, device=device)
    poisson_dist = torch.distributions.Poisson(poisson_rate)
    spike_counts_gpu = torch.zeros(steps, dtype=torch.int64, device=device)

    torch.matmul(prev_spikes, w)
    torch.cuda.synchronize()

    gc.collect()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()

    for t in range(steps):
        ext = poisson_dist.sample((n,))
        I_syn = torch.matmul(prev_spikes, w)
        v += ext * cfg.weight_exc + I_syn
        v += alpha * (cfg.v_rest - v)
        fired = v >= cfg.v_threshold
        spike_counts_gpu[t] = fired.sum()
        v = torch.where(fired, v_reset_t, v)
        prev_spikes = fired.float()

    torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    gpu_mem_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

    step_counts_cpu = spike_counts_gpu.cpu().tolist()
    spike_count = int(spike_counts_gpu.sum().item())

    m = _compute_extended_metrics(spike_count, n, cfg.sim_ms, n_synapses, wall, step_counts_cpu)
    return RunMetrics(
        wall_time_s=wall,
        peak_rss_mb=gpu_mem_mb,
        total_spikes=spike_count,
        mean_rate_hz=m["rate"],
        synaptic_events=m["syn_events"],
        synaptic_events_per_s=m["syn_per_s"],
        activation_sparsity=m["sparsity"],
        gpu_mem_mb=gpu_mem_mb,
    )


# ---------------------------------------------------------------------------
# Simulator: PyTorch CUDA (sparse CSR)
# ---------------------------------------------------------------------------
def run_pytorch_cuda_sparse(cfg: BrunelConfig) -> RunMetrics | None:
    try:
        import torch

        if not torch.cuda.is_available():
            return None
    except ImportError:
        return None

    device = torch.device("cuda")
    rng = np.random.default_rng(cfg.seed)
    n = cfg.n_neurons
    weights_np, n_synapses = _build_weights_dense(cfg, rng)

    from scipy import sparse as sp

    w_coo = sp.coo_matrix(weights_np.T)
    indices = np.vstack([w_coo.row, w_coo.col])
    w_sparse = torch.sparse_coo_tensor(
        torch.tensor(indices, dtype=torch.long, device=device),
        torch.tensor(w_coo.data, dtype=torch.float32, device=device),
        size=(n, n),
    ).to_sparse_csr()
    del weights_np

    v = torch.full((n,), cfg.v_rest, dtype=torch.float32, device=device)
    alpha = cfg.dt / cfg.tau_mem
    steps = int(cfg.sim_ms / cfg.dt)
    prev_spikes = torch.zeros(n, dtype=torch.float32, device=device)

    v_reset_t = torch.full((n,), cfg.v_reset, dtype=torch.float32, device=device)
    poisson_rate = torch.tensor(cfg.ext_poisson_lambda, device=device)
    poisson_dist = torch.distributions.Poisson(poisson_rate)
    spike_counts_gpu = torch.zeros(steps, dtype=torch.int64, device=device)

    # warmup
    torch.sparse.mm(w_sparse, prev_spikes.unsqueeze(1))
    torch.cuda.synchronize()

    gc.collect()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()

    for t in range(steps):
        ext = poisson_dist.sample((n,))
        I_syn = torch.sparse.mm(w_sparse, prev_spikes.unsqueeze(1)).squeeze(1)
        v += ext * cfg.weight_exc + I_syn
        v += alpha * (cfg.v_rest - v)
        fired = v >= cfg.v_threshold
        spike_counts_gpu[t] = fired.sum()
        v = torch.where(fired, v_reset_t, v)
        prev_spikes = fired.float()

    torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    gpu_mem_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

    step_counts_cpu = spike_counts_gpu.cpu().tolist()
    spike_count = int(spike_counts_gpu.sum().item())

    m = _compute_extended_metrics(spike_count, n, cfg.sim_ms, n_synapses, wall, step_counts_cpu)
    return RunMetrics(
        wall_time_s=wall,
        peak_rss_mb=gpu_mem_mb,
        total_spikes=spike_count,
        mean_rate_hz=m["rate"],
        synaptic_events=m["syn_events"],
        synaptic_events_per_s=m["syn_per_s"],
        activation_sparsity=m["sparsity"],
        gpu_mem_mb=gpu_mem_mb,
    )


# ---------------------------------------------------------------------------
# Simulator: Brian2 (with JIT warmup)
# ---------------------------------------------------------------------------
_brian2_warmed_up = False


def _brian2_warmup():
    global _brian2_warmed_up
    if _brian2_warmed_up:
        return
    import brian2

    brian2.start_scope()
    G = brian2.NeuronGroup(
        100, "dv/dt = -v/(20*ms) : 1", threshold="v>20", reset="v=10", method="euler"
    )
    G.v = 0
    brian2.run(10 * brian2.ms)
    _brian2_warmed_up = True


def run_brian2(cfg: BrunelConfig) -> RunMetrics | None:
    try:
        import brian2
    except ImportError:
        return None

    _brian2_warmup()
    brian2.start_scope()

    eqs = "dv/dt = -v / (tau * ms) : 1\ntau : 1"
    G = brian2.NeuronGroup(
        cfg.n_neurons,
        eqs,
        threshold="v > v_th",
        reset="v = v_reset",
        method="euler",
        dt=cfg.dt * brian2.ms,
    )
    G.v = 0
    G.tau = cfg.tau_mem
    G.namespace["v_th"] = cfg.v_threshold
    G.namespace["v_reset"] = cfg.v_reset

    S_exc = brian2.Synapses(G[: cfg.n_exc], G, on_pre="v_post += w", dt=cfg.dt * brian2.ms)
    S_exc.connect(p=cfg.conn_prob)
    S_exc.namespace["w"] = cfg.weight_exc

    S_inh = brian2.Synapses(G[cfg.n_exc :], G, on_pre="v_post -= w", dt=cfg.dt * brian2.ms)
    S_inh.connect(p=cfg.conn_prob)
    S_inh.namespace["w"] = cfg.weight_inh

    # Independent Poisson input per neuron: each neuron draws from its own
    # Poisson process with c_ext sources at external_rate_hz each.
    # PoissonInput (not PoissonGroup+connect) avoids shared-source correlation.
    c_ext_int = max(1, int(cfg.c_ext))
    P_ext = brian2.PoissonInput(  # noqa: F841
        G, "v", N=c_ext_int, rate=cfg.external_rate_hz * brian2.Hz, weight=cfg.weight_exc
    )

    mon = brian2.SpikeMonitor(G)
    n_synapses = int(cfg.n_neurons * cfg.n_neurons * cfg.conn_prob)

    gc.collect()
    tracemalloc.start()
    t0 = time.perf_counter()
    brian2.run(cfg.sim_ms * brian2.ms)
    wall = time.perf_counter() - t0
    peak_mb = _tracemalloc_peak_mb()
    tracemalloc.stop()

    n_spikes = mon.num_spikes
    rate = n_spikes / (cfg.sim_ms / 1000.0) / cfg.n_neurons
    fan_out = n_synapses / cfg.n_neurons if cfg.n_neurons > 0 else 0
    syn_events = int(n_spikes * fan_out)

    return RunMetrics(
        wall_time_s=wall,
        peak_rss_mb=peak_mb,
        total_spikes=n_spikes,
        mean_rate_hz=rate,
        synaptic_events=syn_events,
        synaptic_events_per_s=syn_events / wall if wall > 0 else 0.0,
        activation_sparsity=0.0,  # Brian2 doesn't expose per-step counts easily
    )


# ---------------------------------------------------------------------------
# Simulator: NEST
# ---------------------------------------------------------------------------
def run_nest(cfg: BrunelConfig) -> RunMetrics | None:
    try:
        import nest
    except ImportError:
        return None

    nest.ResetKernel()
    nest.set(resolution=cfg.dt, rng_seed=cfg.seed)

    neurons = nest.Create(
        "iaf_psc_delta",
        cfg.n_neurons,
        params={
            "V_th": cfg.v_threshold,
            "V_reset": cfg.v_reset,
            "E_L": cfg.v_rest,
            "V_m": cfg.v_rest,
            "tau_m": cfg.tau_mem,
            "t_ref": 0.0,
            "C_m": cfg.tau_mem,
        },
    )
    exc = neurons[: cfg.n_exc]
    inh = neurons[cfg.n_exc :]

    nest.Connect(
        exc,
        neurons,
        conn_spec={"rule": "pairwise_bernoulli", "p": cfg.conn_prob},
        syn_spec={"weight": cfg.weight_exc, "delay": cfg.dt},
    )
    nest.Connect(
        inh,
        neurons,
        conn_spec={"rule": "pairwise_bernoulli", "p": cfg.conn_prob},
        syn_spec={"weight": -cfg.weight_inh, "delay": cfg.dt},
    )

    poisson = nest.Create("poisson_generator", params={"rate": cfg.c_ext * cfg.external_rate_hz})
    nest.Connect(
        poisson,
        neurons,
        conn_spec={"rule": "all_to_all"},
        syn_spec={"weight": cfg.weight_exc, "delay": cfg.dt},
    )

    sr = nest.Create("spike_recorder")
    nest.Connect(neurons, sr)

    n_synapses = int(cfg.n_neurons * cfg.n_neurons * cfg.conn_prob)

    gc.collect()
    tracemalloc.start()
    t0 = time.perf_counter()
    nest.Simulate(cfg.sim_ms)
    wall = time.perf_counter() - t0
    peak_mb = _tracemalloc_peak_mb()
    tracemalloc.stop()

    events = sr.get("events")
    n_spikes = len(events["senders"])
    rate = n_spikes / (cfg.sim_ms / 1000.0) / cfg.n_neurons
    fan_out = n_synapses / cfg.n_neurons if cfg.n_neurons > 0 else 0
    syn_events = int(n_spikes * fan_out)

    return RunMetrics(
        wall_time_s=wall,
        peak_rss_mb=peak_mb,
        total_spikes=n_spikes,
        mean_rate_hz=rate,
        synaptic_events=syn_events,
        synaptic_events_per_s=syn_events / wall if wall > 0 else 0.0,
        activation_sparsity=0.0,
    )


# ---------------------------------------------------------------------------
# Simulator: Rust engine (batch_lif_run_multi)
# ---------------------------------------------------------------------------
def run_rust_engine(cfg: BrunelConfig) -> RunMetrics | None:
    """Run the unconnected fixed-point LIF layer on the Rust engine.

    This lane has no recurrent connectivity: each neuron receives a constant
    external drive plus a per-neuron noise offset, so its firing rate does not
    track the recurrent Brunel network and its rows are NOT a dynamics-parity
    comparison against Brian2. The gate in
    ``tests/test_brunel_dynamics_parity_gate.py`` pins this boundary.
    """
    try:
        import sc_neurocore_engine as eng
    except ImportError:
        return None

    n = cfg.n_neurons
    steps = int(cfg.sim_ms / cfg.dt)

    # Q8.8 fixed-point parameters matching the Rust FixedPointLIF
    fraction = 8
    scale = 1 << fraction
    v_th_fp = int(cfg.v_threshold * scale)
    v_reset_fp = int(cfg.v_reset * scale)
    v_rest_fp = int(cfg.v_rest * scale)

    # Leak: alpha = dt/tau, leak_k = int(alpha * 256) in Q8.8
    alpha = cfg.dt / cfg.tau_mem
    leak_k = max(1, int(alpha * scale))

    # Input current: mean external drive in Q8.8
    mean_ext = cfg.ext_poisson_lambda * cfg.weight_exc
    i_t_fp = int(mean_ext * scale)
    gain_k = scale  # unity gain

    # Noise proportional to Poisson variance
    noise_fp = max(1, int(np.sqrt(cfg.ext_poisson_lambda) * cfg.weight_exc * scale * 0.5))

    gc.collect()
    tracemalloc.start()
    t0 = time.perf_counter()

    # Parallel LIF: each neuron gets constant input + per-neuron noise offset
    rng = np.random.default_rng(cfg.seed)
    noise_offsets = rng.normal(0, noise_fp, n).astype(np.int16)
    currents = (np.full(n, i_t_fp, dtype=np.int16) + noise_offsets).astype(np.int16)
    refractory_steps = (
        max(0, int(cfg.refractory_ms / cfg.dt)) if hasattr(cfg, "refractory_ms") else 2
    )
    spikes_arr, _ = eng.batch_lif_run_multi(
        n,
        steps,
        leak_k,
        gain_k,
        currents,
        16,  # data_width
        fraction,
        v_rest_fp,
        v_reset_fp,
        v_th_fp,
        refractory_steps,
    )

    wall = time.perf_counter() - t0
    peak_mb = _tracemalloc_peak_mb()
    tracemalloc.stop()

    spikes_np = np.array(spikes_arr, dtype=np.int32)
    total_spikes = int(spikes_np.sum())
    rate = total_spikes / (cfg.sim_ms / 1000.0) / n if n > 0 else 0.0

    # No recurrent connectivity — syn_events from external drive only
    syn_events = int(total_spikes * cfg.c_ext)

    step_counts = spikes_np.sum(axis=0).tolist() if spikes_np.ndim == 2 else []
    active_per_step = [int(c > 0) for c in step_counts] if step_counts else []
    sparsity = (
        1.0 - (sum(active_per_step) / len(active_per_step) / n)
        if active_per_step and n > 0
        else 0.0
    )

    return RunMetrics(
        wall_time_s=wall,
        peak_rss_mb=peak_mb,
        total_spikes=total_spikes,
        mean_rate_hz=rate,
        synaptic_events=syn_events,
        synaptic_events_per_s=syn_events / wall if wall > 0 else 0.0,
        activation_sparsity=sparsity,
    )


# ---------------------------------------------------------------------------
# Simulator: Rust Brunel (fused CSR spike-scatter network)
# ---------------------------------------------------------------------------
def run_rust_brunel(cfg: BrunelConfig) -> RunMetrics | None:
    try:
        import sc_neurocore_engine as eng
    except ImportError:
        return None

    if not hasattr(eng, "BrunelNetwork"):
        return None

    from scipy import sparse as sp

    rng = np.random.default_rng(cfg.seed)
    n = cfg.n_neurons
    weights_np, n_synapses = _build_weights_dense(cfg, rng)

    # Q8.8 fixed-point
    fraction = 8
    scale = 1 << fraction
    w_csr = sp.csr_matrix(weights_np)
    w_data_fp = np.clip(w_csr.data * scale, -32768, 32767).astype(np.int16)

    alpha = cfg.dt / cfg.tau_mem
    leak_k = max(1, int(alpha * scale))
    gain_k = scale
    ext_weight_fp = int(cfg.weight_exc * scale)
    steps = int(cfg.sim_ms / cfg.dt)

    gc.collect()
    tracemalloc.start()
    t0 = time.perf_counter()

    net = eng.BrunelNetwork(
        n_neurons=n,
        w_indptr=w_csr.indptr.astype(np.int64),
        w_indices=w_csr.indices.astype(np.int64),
        w_data=w_data_fp,
        leak_k=leak_k,
        gain_k=gain_k,
        ext_lambda=cfg.ext_poisson_lambda,
        ext_weight_fp=ext_weight_fp,
        data_width=16,
        fraction=fraction,
        v_rest=int(cfg.v_rest * scale),
        v_reset=int(cfg.v_reset * scale),
        v_threshold=int(cfg.v_threshold * scale),
        refractory_period=2,
        seed=cfg.seed,
    )
    counts = np.asarray(net.run(steps))

    wall = time.perf_counter() - t0
    peak_mb = _tracemalloc_peak_mb()
    tracemalloc.stop()

    spike_count = int(counts.sum())
    step_counts = counts.tolist()
    m = _compute_extended_metrics(spike_count, n, cfg.sim_ms, n_synapses, wall, step_counts)
    return RunMetrics(
        wall_time_s=wall,
        peak_rss_mb=peak_mb,
        total_spikes=spike_count,
        mean_rate_hz=m["rate"],
        synaptic_events=m["syn_events"],
        synaptic_events_per_s=m["syn_per_s"],
        activation_sparsity=m["sparsity"],
    )


# ---------------------------------------------------------------------------
# Simulator: Norse (PyTorch-based SNN)
# ---------------------------------------------------------------------------
def run_norse(cfg: BrunelConfig) -> RunMetrics | None:
    try:
        import torch
        import norse.torch as norse  # noqa: F811

        if not torch.cuda.is_available():
            return None
    except ImportError:
        return None

    device = torch.device("cuda")
    rng = np.random.default_rng(cfg.seed)
    n = cfg.n_neurons
    weights_np, n_synapses = _build_weights_dense(cfg, rng)
    steps = int(cfg.sim_ms / cfg.dt)

    w = torch.tensor(weights_np, dtype=torch.float32, device=device)
    tau_mem_inv = 1.0 / (cfg.tau_mem * 1e-3)  # Norse uses inverse tau in seconds
    p = norse.LIFParameters(
        tau_mem_inv=torch.tensor(tau_mem_inv),
        v_th=torch.tensor(cfg.v_threshold),
        v_reset=torch.tensor(cfg.v_reset),
        v_leak=torch.tensor(cfg.v_rest),
    )
    state = norse.LIFState(
        v=torch.full((n,), cfg.v_rest, device=device),
        i=torch.zeros(n, device=device),
    )

    poisson_dist = torch.distributions.Poisson(torch.tensor(cfg.ext_poisson_lambda, device=device))
    spike_counts_gpu = torch.zeros(steps, dtype=torch.int64, device=device)
    torch.cuda.synchronize()

    gc.collect()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()

    prev_z = torch.zeros(n, device=device)
    for t in range(steps):
        ext = poisson_dist.sample((n,)) * cfg.weight_exc
        i_syn = torch.matmul(prev_z, w) + ext
        z, state = norse.lif_step(i_syn, state, p=p, dt=cfg.dt * 1e-3)
        spike_counts_gpu[t] = z.sum()
        prev_z = z

    torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    gpu_mem_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

    step_counts_cpu = spike_counts_gpu.cpu().tolist()
    spike_count = int(spike_counts_gpu.sum().item())
    m = _compute_extended_metrics(spike_count, n, cfg.sim_ms, n_synapses, wall, step_counts_cpu)
    return RunMetrics(
        wall_time_s=wall,
        peak_rss_mb=gpu_mem_mb,
        total_spikes=spike_count,
        mean_rate_hz=m["rate"],
        synaptic_events=m["syn_events"],
        synaptic_events_per_s=m["syn_per_s"],
        activation_sparsity=m["sparsity"],
        gpu_mem_mb=gpu_mem_mb,
    )


# ---------------------------------------------------------------------------
# Simulator: snnTorch (PyTorch-based SNN)
# ---------------------------------------------------------------------------
def run_snntorch(cfg: BrunelConfig) -> RunMetrics | None:
    try:
        import torch
        import snntorch as snn

        if not torch.cuda.is_available():
            return None
    except ImportError:
        return None

    device = torch.device("cuda")
    rng = np.random.default_rng(cfg.seed)
    n = cfg.n_neurons
    weights_np, n_synapses = _build_weights_dense(cfg, rng)
    steps = int(cfg.sim_ms / cfg.dt)

    w = torch.tensor(weights_np, dtype=torch.float32, device=device)
    beta = float(np.exp(-cfg.dt / cfg.tau_mem))
    lif = snn.Leaky(
        beta=beta,
        threshold=cfg.v_threshold,
        reset_mechanism="zero",
        init_hidden=False,
    )

    mem = torch.full((n,), cfg.v_rest, dtype=torch.float32, device=device)
    prev_spk = torch.zeros(n, dtype=torch.float32, device=device)

    poisson_dist = torch.distributions.Poisson(torch.tensor(cfg.ext_poisson_lambda, device=device))
    spike_counts_gpu = torch.zeros(steps, dtype=torch.int64, device=device)
    torch.cuda.synchronize()

    gc.collect()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()

    for t in range(steps):
        ext = poisson_dist.sample((n,)) * cfg.weight_exc
        i_syn = torch.matmul(prev_spk, w) + ext
        spk, mem = lif(i_syn, mem)
        spike_counts_gpu[t] = spk.sum()
        prev_spk = spk

    torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    gpu_mem_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

    step_counts_cpu = spike_counts_gpu.cpu().tolist()
    spike_count = int(spike_counts_gpu.sum().item())
    m = _compute_extended_metrics(spike_count, n, cfg.sim_ms, n_synapses, wall, step_counts_cpu)
    return RunMetrics(
        wall_time_s=wall,
        peak_rss_mb=gpu_mem_mb,
        total_spikes=spike_count,
        mean_rate_hz=m["rate"],
        synaptic_events=m["syn_events"],
        synaptic_events_per_s=m["syn_per_s"],
        activation_sparsity=m["sparsity"],
        gpu_mem_mb=gpu_mem_mb,
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
SIMULATORS = {
    "sc_numpy_dense": ("SC-NeuroCore (NumPy dense)", run_numpy),
    "sc_numpy_sparse": ("SC-NeuroCore (NumPy sparse)", run_numpy_sparse),
    "sc_rust_engine": ("SC-NeuroCore (Rust engine)", run_rust_engine),
    "rust_brunel": ("SC-NeuroCore (Rust Brunel)", run_rust_brunel),
    "sc_pytorch_cuda": ("SC-NeuroCore (PyTorch CUDA)", run_pytorch_cuda),
    "sc_pytorch_cuda_sparse": ("SC-NeuroCore (PyTorch CUDA sparse)", run_pytorch_cuda_sparse),
    "norse": ("Norse (PyTorch SNN)", run_norse),
    "snntorch": ("snnTorch (PyTorch SNN)", run_snntorch),
    "brian2": ("Brian2", run_brian2),
    "nest": ("NEST", run_nest),
}


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def run_scaling(
    scales: list[int],
    sim_ms: float,
    repeats: int,
    regimes: list[str],
    simulators: list[str] | None = None,
) -> list[ScalePoint]:
    targets = simulators or list(SIMULATORS.keys())
    results: list[ScalePoint] = []

    for regime in regimes:
        rinfo = BRUNEL_REGIMES[regime]
        print(f"\n{'#' * 70}")
        print(f"  Regime: {regime} — {rinfo['label']}  (g={rinfo['g_inh']}, eta={rinfo['eta']})")
        print(f"{'#' * 70}")

        for n_neurons in scales:
            cfg = BrunelConfig(n_neurons=n_neurons, regime=regime, sim_ms=sim_ms)
            n_synapses = int(n_neurons * n_neurons * cfg.conn_prob)
            print(
                f"\n  N={n_neurons:,}  synapses~{n_synapses:,}  ext_rate={cfg.external_rate_hz:.1f} Hz"
            )

            for sim_key in targets:
                label, fn = SIMULATORS[sim_key]
                # Skip dense paths if weight matrix > 8 GB
                mem_est_gb = n_neurons * n_neurons * 4 / 1e9
                if sim_key in ("sc_numpy_dense", "sc_pytorch_cuda") and mem_est_gb > 8.0:
                    print(f"    {label}: SKIP (matrix ~{mem_est_gb:.1f} GB)")
                    continue

                sp = ScalePoint(
                    n_neurons=n_neurons, simulator=sim_key, regime=regime, n_synapses=n_synapses
                )
                print(f"    {label}...", end=" ", flush=True)

                for _ in range(repeats):
                    gc.collect()
                    try:
                        result = fn(cfg)
                    except Exception as e:
                        print(f"ERROR: {e}")
                        break
                    if result is None:
                        print("SKIP (not installed)")
                        break
                    sp.runs.append(result)

                if sp.runs:
                    print(
                        f"{sp.wall_mean:.3f}s ±{sp.wall_std:.3f}s  "
                        f"rate={sp.rate_mean:.1f} Hz  "
                        f"syn/s={sp.syn_events_per_s_mean:.2e}  "
                        f"sparsity={sp.sparsity_mean:.3f}  "
                        f"({len(sp.runs)} runs)"
                    )
                    results.append(sp)

    return results


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def _serialize(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    raise TypeError(f"{type(obj).__name__} not serializable")


def to_json(results: list[ScalePoint], sys_info: dict, params: dict) -> dict:
    data = []
    for sp in results:
        data.append(
            {
                "n_neurons": sp.n_neurons,
                "simulator": sp.simulator,
                "regime": sp.regime,
                "n_synapses": sp.n_synapses,
                "wall_mean_s": round(sp.wall_mean, 4),
                "wall_std_s": round(sp.wall_std, 4),
                "wall_min_s": round(sp.wall_min, 4),
                "peak_rss_mb": round(sp.peak_rss_mb, 1),
                "spikes_mean": round(sp.spikes_mean, 1),
                "rate_mean_hz": round(sp.rate_mean, 2),
                "syn_events_per_s": round(sp.syn_events_per_s_mean, 2),
                "activation_sparsity": round(sp.sparsity_mean, 4),
                "n_runs": len(sp.runs),
                "runs": [asdict(r) for r in sp.runs],
            }
        )
    return {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "system": sys_info,
        "params": params,
        "data": data,
    }


def format_markdown(results: list[ScalePoint]) -> str:
    lines = [
        "# SC-NeuroCore Scaling Benchmark",
        "",
        "Brunel balanced network: 80/20 exc/inh, 10% connectivity.",
        "Regimes per Brunel (2000): SR, SI, AI, AR.",
        "",
        "## Results",
        "",
        "| Regime | N | Simulator | Time (s) | Rate (Hz) | Syn events/s | Sparsity | RSS (MB) |",
        "|--------|--:|-----------|----------:|----------:|-------------:|---------:|---------:|",
    ]
    for sp in results:
        lines.append(
            f"| {sp.regime:>6} | {sp.n_neurons:>7,} | {sp.simulator:<28s} "
            f"| {sp.wall_mean:>8.3f} | {sp.rate_mean:>9.1f} "
            f"| {sp.syn_events_per_s_mean:>12.2e} | {sp.sparsity_mean:>8.3f} "
            f"| {sp.peak_rss_mb:>8.1f} |"
        )

    # Speedup vs Brian2
    brian2_times: dict[tuple[str, int], float] = {}
    for sp in results:
        if sp.simulator == "brian2":
            brian2_times[(sp.regime, sp.n_neurons)] = sp.wall_mean

    brian2_rates: dict[tuple[str, int], float] = {}
    for sp in results:
        if sp.simulator == "brian2":
            brian2_rates[(sp.regime, sp.n_neurons)] = sp.rate_mean

    if brian2_times:
        lines.extend(
            [
                "",
                "## Speedup vs Brian2",
                "",
                "Rate ratio = SC rate / Brian2 rate. Speedup only valid when rates match (0.75-1.25x).",
                "",
                "| Regime | N | Simulator | Rate ratio | Speedup | Valid |",
                "|--------|--:|-----------|----------:|--------:|:-----:|",
            ]
        )
        for sp in results:
            if sp.simulator == "brian2":
                continue
            key = (sp.regime, sp.n_neurons)
            b2 = brian2_times.get(key)
            b2r = brian2_rates.get(key, 0.0)
            if b2 and b2 > 0:
                ratio = sp.rate_mean / b2r if b2r > 0 else float("inf")
                valid = 0.75 <= ratio <= 1.25
                speedup = b2 / sp.wall_mean
                mark = "Y" if valid else "N"
                lines.append(
                    f"| {sp.regime} | {sp.n_neurons:>7,} | {sp.simulator:<28s} "
                    f"| {ratio:>9.2f}x | {speedup:>7.1f}x | {mark:>5s} |"
                )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Scaling benchmark: SC-NeuroCore vs Brian2 vs NEST")
    ap.add_argument(
        "--scales", type=int, nargs="+", default=[1000, 2000, 5000, 10000, 20000, 50000]
    )
    ap.add_argument("--sim-ms", type=float, default=500.0)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--regimes", nargs="+", choices=list(BRUNEL_REGIMES.keys()), default=["AI"])
    ap.add_argument("--simulators", nargs="+", choices=list(SIMULATORS.keys()))
    ap.add_argument("--json", type=str)
    ap.add_argument("--markdown", action="store_true")
    ap.add_argument("--no-gpu", action="store_true")
    args = ap.parse_args()

    sims = args.simulators
    if args.no_gpu and sims is None:
        sims = [k for k in SIMULATORS if "cuda" not in k]

    sys_info = _system_info()
    params = {
        "sim_ms": args.sim_ms,
        "repeats": args.repeats,
        "regimes": args.regimes,
        "conn_prob": 0.1,
        "weight_exc": 0.1,
    }

    print("=" * 70)
    print("  SC-NeuroCore Scaling Benchmark (4-Regime Brunel)")
    print(f"  Scales: {args.scales}")
    print(f"  Regimes: {args.regimes}")
    print(f"  Sim: {args.sim_ms} ms, Repeats: {args.repeats}")
    print(f"  System: {sys_info.get('cpu', 'unknown')}")
    if "gpu" in sys_info:
        print(f"  GPU: {sys_info['gpu']}")
    print("=" * 70)

    results = run_scaling(args.scales, args.sim_ms, args.repeats, args.regimes, sims)

    if args.markdown:
        print("\n" + format_markdown(results))

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        out = to_json(results, sys_info, params)
        Path(args.json).write_text(json.dumps(out, indent=2, default=_serialize))
        print(f"\nResults written to {args.json}")

    if not args.json and not args.markdown:
        print("\n" + format_markdown(results))


if __name__ == "__main__":
    main()

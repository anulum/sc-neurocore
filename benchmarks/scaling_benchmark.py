# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Scaling Benchmark — SC-NeuroCore vs Brian2 vs NEST
===================================================

Measures wall-clock time, memory footprint, and spike statistics as neuron
count scales from 1K to 50K on the Brunel balanced network (80% exc, 20% inh,
10% connectivity, Poisson drive).

Produces JSON + markdown suitable for paper figures (scaling curves, memory
plots, latency histograms).

Usage::

    python benchmarks/scaling_benchmark.py                    # default scales
    python benchmarks/scaling_benchmark.py --scales 1000 5000 10000
    python benchmarks/scaling_benchmark.py --repeats 5 --sim-ms 500
    python benchmarks/scaling_benchmark.py --json results.json --markdown

Requires: numpy, brian2 (optional), nest (optional), torch (optional)
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

try:
    import resource as _resource
except ImportError:
    _resource = None  # Windows

import numpy as np


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class RunMetrics:
    """Single run measurement."""
    wall_time_s: float
    peak_rss_mb: float
    total_spikes: int
    mean_rate_hz: float


@dataclass
class ScalePoint:
    """Aggregated metrics at one neuron count."""
    n_neurons: int
    simulator: str
    n_synapses: int
    runs: list[RunMetrics] = field(default_factory=list)

    @property
    def wall_mean(self) -> float:
        return float(np.mean([r.wall_time_s for r in self.runs]))

    @property
    def wall_std(self) -> float:
        return float(np.std([r.wall_time_s for r in self.runs], ddof=1)) if len(self.runs) > 1 else 0.0

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


@dataclass
class BenchmarkResult:
    """Complete benchmark output."""
    timestamp: str
    system: dict
    params: dict
    scales: list[int]
    data: list[dict]


# ---------------------------------------------------------------------------
# Memory measurement
# ---------------------------------------------------------------------------
def _get_rss_mb() -> float:
    """Peak RSS in MB. Uses resource module (Linux/macOS) or psutil fallback."""
    if _resource is not None:
        try:
            usage = _resource.getrusage(_resource.RUSAGE_SELF)
            if sys.platform == "darwin":
                return usage.ru_maxrss / 1024 / 1024
            return usage.ru_maxrss / 1024
        except Exception:
            pass
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1024 / 1024
    except ImportError:
        return 0.0


def _measure_rss_delta(fn) -> tuple:
    """Run fn(), return (result, delta_rss_mb). Forces GC before/after."""
    gc.collect()
    rss_before = _get_rss_mb()
    result = fn()
    gc.collect()
    rss_after = _get_rss_mb()
    return result, max(0.0, rss_after - rss_before)


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
# Brunel network — shared parameters
# ---------------------------------------------------------------------------
@dataclass
class BrunelConfig:
    n_neurons: int
    sim_ms: float = 500.0
    dt: float = 0.1
    conn_prob: float = 0.1
    weight_exc: float = 0.1
    g_inh: float = 5.0
    v_threshold: float = 20.0
    v_reset: float = 10.0
    v_rest: float = 0.0
    tau_mem: float = 20.0
    external_rate_hz: float = 20.0
    seed: int = 42

    @property
    def n_exc(self) -> int:
        return int(self.n_neurons * 0.8)

    @property
    def n_inh(self) -> int:
        return self.n_neurons - self.n_exc

    @property
    def weight_inh(self) -> float:
        return self.g_inh * self.weight_exc


# ---------------------------------------------------------------------------
# Simulator: Vectorized NumPy (SC-NeuroCore V20)
# ---------------------------------------------------------------------------
def run_numpy(cfg: BrunelConfig) -> RunMetrics:
    """Fully vectorized NumPy Brunel. O(N²) weight matrix, O(N) neuron update."""
    rng = np.random.default_rng(cfg.seed)
    n = cfg.n_neurons

    conn_mask = rng.random((n, n)) < cfg.conn_prob
    np.fill_diagonal(conn_mask, False)
    weights = np.where(conn_mask, cfg.weight_exc, 0.0).astype(np.float32)
    weights[cfg.n_exc:, :] *= -cfg.g_inh
    n_synapses = int(conn_mask.sum())

    v = np.full(n, cfg.v_rest, dtype=np.float32)
    alpha = np.float32(cfg.dt / cfg.tau_mem)
    steps = int(cfg.sim_ms / cfg.dt)
    spike_count = 0
    prev_spikes = np.zeros(n, dtype=bool)

    gc.collect()
    rss_before = _get_rss_mb()
    t0 = time.perf_counter()

    for _ in range(steps):
        ext = rng.poisson(cfg.external_rate_hz * cfg.dt / 1000.0, n).astype(np.float32)
        I_syn = weights[prev_spikes].sum(axis=0) if prev_spikes.any() else np.zeros(n, dtype=np.float32)
        v += ext * cfg.weight_exc + I_syn
        v += alpha * (cfg.v_rest - v)
        fired = v >= cfg.v_threshold
        spike_count += int(fired.sum())
        v[fired] = cfg.v_reset
        prev_spikes = fired

    wall = time.perf_counter() - t0
    rss_after = _get_rss_mb()
    rate = spike_count / (cfg.sim_ms / 1000.0) / n

    return RunMetrics(
        wall_time_s=wall,
        peak_rss_mb=max(0.0, rss_after - rss_before),
        total_spikes=spike_count,
        mean_rate_hz=rate,
    )


# ---------------------------------------------------------------------------
# Simulator: Sparse NumPy (SC-NeuroCore V20-sparse)
# ---------------------------------------------------------------------------
def run_numpy_sparse(cfg: BrunelConfig) -> RunMetrics:
    """Sparse CSR weight matrix — O(nnz) per step instead of O(N²)."""
    from scipy import sparse

    rng = np.random.default_rng(cfg.seed)
    n = cfg.n_neurons

    conn_mask = rng.random((n, n)) < cfg.conn_prob
    np.fill_diagonal(conn_mask, False)
    w_dense = np.where(conn_mask, cfg.weight_exc, 0.0).astype(np.float32)
    w_dense[cfg.n_exc:, :] *= -cfg.g_inh
    # CSR: rows = presynaptic, cols = postsynaptic → w[pre, post]
    # To get I_syn[post] = sum over pre that fired: w[fired, :].sum(0)
    # With CSR, extracting rows of fired neurons is efficient.
    w_csr = sparse.csr_matrix(w_dense)
    n_synapses = w_csr.nnz
    del w_dense, conn_mask

    v = np.full(n, cfg.v_rest, dtype=np.float32)
    alpha = np.float32(cfg.dt / cfg.tau_mem)
    steps = int(cfg.sim_ms / cfg.dt)
    spike_count = 0
    fired_indices = np.array([], dtype=np.intp)

    gc.collect()
    rss_before = _get_rss_mb()
    t0 = time.perf_counter()

    for _ in range(steps):
        ext = rng.poisson(cfg.external_rate_hz * cfg.dt / 1000.0, n).astype(np.float32)
        if fired_indices.size > 0:
            I_syn = np.asarray(w_csr[fired_indices].sum(axis=0)).ravel()
        else:
            I_syn = np.zeros(n, dtype=np.float32)
        v += ext * cfg.weight_exc + I_syn
        v += alpha * (cfg.v_rest - v)
        fired_mask = v >= cfg.v_threshold
        spike_count += int(fired_mask.sum())
        v[fired_mask] = cfg.v_reset
        fired_indices = np.nonzero(fired_mask)[0]

    wall = time.perf_counter() - t0
    rss_after = _get_rss_mb()
    rate = spike_count / (cfg.sim_ms / 1000.0) / n

    return RunMetrics(
        wall_time_s=wall,
        peak_rss_mb=max(0.0, rss_after - rss_before),
        total_spikes=spike_count,
        mean_rate_hz=rate,
    )


# ---------------------------------------------------------------------------
# Simulator: PyTorch CUDA (SC-NeuroCore V19)
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

    conn_mask = rng.random((n, n)) < cfg.conn_prob
    np.fill_diagonal(conn_mask, False)
    w_np = np.where(conn_mask, cfg.weight_exc, 0.0).astype(np.float32)
    w_np[cfg.n_exc:, :] *= -cfg.g_inh
    n_synapses = int(conn_mask.sum())

    w = torch.tensor(w_np, dtype=torch.float32, device=device)
    v = torch.full((n,), cfg.v_rest, dtype=torch.float32, device=device)
    alpha = cfg.dt / cfg.tau_mem
    steps = int(cfg.sim_ms / cfg.dt)
    spike_count = 0
    prev_spikes = torch.zeros(n, dtype=torch.float32, device=device)

    # Warm-up GPU
    torch.matmul(prev_spikes, w)
    torch.cuda.synchronize()

    gc.collect()
    torch.cuda.reset_peak_memory_stats()
    rss_before = _get_rss_mb()
    t0 = time.perf_counter()

    for _ in range(steps):
        ext = torch.tensor(
            rng.poisson(cfg.external_rate_hz * cfg.dt / 1000.0, n),
            dtype=torch.float32, device=device,
        )
        I_syn = torch.matmul(prev_spikes, w)
        v += ext * cfg.weight_exc + I_syn
        v += alpha * (cfg.v_rest - v)
        fired = v >= cfg.v_threshold
        spike_count += int(fired.sum().item())
        v = torch.where(fired, torch.tensor(cfg.v_reset, device=device), v)
        prev_spikes = fired.float()

    torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    rss_after = _get_rss_mb()
    gpu_mem_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
    rate = spike_count / (cfg.sim_ms / 1000.0) / n

    return RunMetrics(
        wall_time_s=wall,
        peak_rss_mb=gpu_mem_mb,  # GPU memory is the interesting metric here
        total_spikes=spike_count,
        mean_rate_hz=rate,
    )


# ---------------------------------------------------------------------------
# Simulator: Brian2
# ---------------------------------------------------------------------------
def run_brian2(cfg: BrunelConfig) -> RunMetrics | None:
    try:
        import brian2
    except ImportError:
        return None

    brian2.start_scope()

    eqs = """
    dv/dt = -v / (tau * ms) : 1
    tau : 1
    """
    G = brian2.NeuronGroup(
        cfg.n_neurons, eqs,
        threshold="v > v_th", reset="v = v_reset",
        method="euler", dt=cfg.dt * brian2.ms,
    )
    G.v = 0
    G.tau = cfg.tau_mem
    G.namespace["v_th"] = cfg.v_threshold
    G.namespace["v_reset"] = cfg.v_reset

    S_exc = brian2.Synapses(G[:cfg.n_exc], G, on_pre="v_post += w", dt=cfg.dt * brian2.ms)
    S_exc.connect(p=cfg.conn_prob)
    S_exc.namespace["w"] = cfg.weight_exc

    S_inh = brian2.Synapses(G[cfg.n_exc:], G, on_pre="v_post -= w", dt=cfg.dt * brian2.ms)
    S_inh.connect(p=cfg.conn_prob)
    S_inh.namespace["w"] = cfg.weight_inh

    P_ext = brian2.PoissonGroup(cfg.n_neurons, rates=cfg.external_rate_hz * brian2.Hz)
    S_ext = brian2.Synapses(P_ext, G, on_pre="v_post += w", dt=cfg.dt * brian2.ms)
    S_ext.connect(j="i")
    S_ext.namespace["w"] = cfg.weight_exc

    mon = brian2.SpikeMonitor(G)

    gc.collect()
    rss_before = _get_rss_mb()
    t0 = time.perf_counter()
    brian2.run(cfg.sim_ms * brian2.ms)
    wall = time.perf_counter() - t0
    rss_after = _get_rss_mb()

    rate = mon.num_spikes / (cfg.sim_ms / 1000.0) / cfg.n_neurons

    return RunMetrics(
        wall_time_s=wall,
        peak_rss_mb=max(0.0, rss_after - rss_before),
        total_spikes=mon.num_spikes,
        mean_rate_hz=rate,
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

    neurons = nest.Create("iaf_psc_delta", cfg.n_neurons, params={
        "V_th": cfg.v_threshold,
        "V_reset": cfg.v_reset,
        "E_L": cfg.v_rest,
        "V_m": cfg.v_rest,
        "tau_m": cfg.tau_mem,
        "t_ref": 0.0,
        "C_m": cfg.tau_mem,  # C_m = tau_m / R; with R=1 → C_m = tau_m
    })

    exc = neurons[:cfg.n_exc]
    inh = neurons[cfg.n_exc:]

    nest.Connect(exc, neurons,
                 conn_spec={"rule": "pairwise_bernoulli", "p": cfg.conn_prob},
                 syn_spec={"weight": cfg.weight_exc, "delay": cfg.dt})
    nest.Connect(inh, neurons,
                 conn_spec={"rule": "pairwise_bernoulli", "p": cfg.conn_prob},
                 syn_spec={"weight": -cfg.weight_inh, "delay": cfg.dt})

    poisson = nest.Create("poisson_generator", params={"rate": cfg.external_rate_hz})
    nest.Connect(poisson, neurons,
                 conn_spec={"rule": "all_to_all"},
                 syn_spec={"weight": cfg.weight_exc, "delay": cfg.dt})

    sr = nest.Create("spike_recorder")
    nest.Connect(neurons, sr)

    gc.collect()
    rss_before = _get_rss_mb()
    t0 = time.perf_counter()
    nest.Simulate(cfg.sim_ms)
    wall = time.perf_counter() - t0
    rss_after = _get_rss_mb()

    events = sr.get("events")
    n_spikes = len(events["senders"])
    rate = n_spikes / (cfg.sim_ms / 1000.0) / cfg.n_neurons

    return RunMetrics(
        wall_time_s=wall,
        peak_rss_mb=max(0.0, rss_after - rss_before),
        total_spikes=n_spikes,
        mean_rate_hz=rate,
    )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
SIMULATORS = {
    "sc_numpy_dense": ("SC-NeuroCore (NumPy dense)", run_numpy),
    "sc_numpy_sparse": ("SC-NeuroCore (NumPy sparse)", run_numpy_sparse),
    "sc_pytorch_cuda": ("SC-NeuroCore (PyTorch CUDA)", run_pytorch_cuda),
    "brian2": ("Brian2", run_brian2),
    "nest": ("NEST", run_nest),
}


def run_scaling(
    scales: list[int],
    sim_ms: float,
    repeats: int,
    simulators: list[str] | None = None,
) -> list[ScalePoint]:
    targets = simulators or list(SIMULATORS.keys())
    results: list[ScalePoint] = []

    for n_neurons in scales:
        cfg = BrunelConfig(n_neurons=n_neurons, sim_ms=sim_ms)
        n_synapses = int(n_neurons * n_neurons * cfg.conn_prob)
        print(f"\n{'='*70}")
        print(f"  N = {n_neurons:,} neurons, ~{n_synapses:,} synapses, sim = {sim_ms} ms")
        print(f"{'='*70}")

        for sim_key in targets:
            label, fn = SIMULATORS[sim_key]
            print(f"  {label}...", end=" ", flush=True)

            # Skip if N² weight matrix would exceed ~8 GB (float32)
            mem_est_gb = n_neurons * n_neurons * 4 / 1e9
            if sim_key in ("sc_numpy_dense", "sc_pytorch_cuda") and mem_est_gb > 8.0:
                print(f"SKIPPED (weight matrix ~{mem_est_gb:.1f} GB)")
                continue

            sp = ScalePoint(n_neurons=n_neurons, simulator=sim_key, n_synapses=n_synapses)

            for rep in range(repeats):
                gc.collect()
                try:
                    result = fn(cfg)
                except Exception as e:
                    print(f"ERROR: {e}")
                    break
                if result is None:
                    print("SKIPPED (not installed)")
                    break
                sp.runs.append(result)

            if sp.runs:
                print(
                    f"{sp.wall_mean:.3f}s ± {sp.wall_std:.3f}s, "
                    f"RSS: {sp.peak_rss_mb:.0f} MB, "
                    f"{sp.rate_mean:.1f} Hz "
                    f"({len(sp.runs)} runs)"
                )
                results.append(sp)

    return results


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------
def _serialize(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    raise TypeError(f"{type(obj).__name__} not serializable")


def to_json(results: list[ScalePoint], sys_info: dict, params: dict, scales: list[int]) -> dict:
    data = []
    for sp in results:
        data.append({
            "n_neurons": sp.n_neurons,
            "simulator": sp.simulator,
            "n_synapses": sp.n_synapses,
            "wall_mean_s": round(sp.wall_mean, 4),
            "wall_std_s": round(sp.wall_std, 4),
            "wall_min_s": round(sp.wall_min, 4),
            "peak_rss_mb": round(sp.peak_rss_mb, 1),
            "spikes_mean": round(sp.spikes_mean, 1),
            "rate_mean_hz": round(sp.rate_mean, 2),
            "n_runs": len(sp.runs),
            "runs": [asdict(r) for r in sp.runs],
        })
    return {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "system": sys_info,
        "params": params,
        "scales": scales,
        "data": data,
    }


def format_markdown(results: list[ScalePoint]) -> str:
    lines = [
        "# SC-NeuroCore Scaling Benchmark",
        "",
        "Brunel balanced network: 80/20 exc/inh, 10% connectivity, Poisson drive 20 Hz.",
        "",
        "## Wall-Clock Time (seconds)",
        "",
        "| N neurons | Simulator | Mean (s) | Std (s) | Min (s) | RSS (MB) | Rate (Hz) |",
        "|----------:|-----------|----------:|--------:|--------:|---------:|----------:|",
    ]

    for sp in results:
        lines.append(
            f"| {sp.n_neurons:>9,} | {sp.simulator:<24s} "
            f"| {sp.wall_mean:>8.3f} | {sp.wall_std:>7.3f} "
            f"| {sp.wall_min:>7.3f} | {sp.peak_rss_mb:>8.0f} "
            f"| {sp.rate_mean:>9.2f} |"
        )

    # Speedup table: SC variants vs Brian2
    lines.extend(["", "## Speedup vs Brian2", ""])
    brian2_by_n = {}
    for sp in results:
        if sp.simulator == "brian2":
            brian2_by_n[sp.n_neurons] = sp.wall_mean

    if brian2_by_n:
        lines.append("| N neurons | Simulator | Speedup vs Brian2 |")
        lines.append("|----------:|-----------|------------------:|")
        for sp in results:
            if sp.simulator == "brian2":
                continue
            b2_time = brian2_by_n.get(sp.n_neurons)
            if b2_time and b2_time > 0:
                speedup = b2_time / sp.wall_mean
                lines.append(f"| {sp.n_neurons:>9,} | {sp.simulator:<24s} | {speedup:>17.1f}x |")

    # Memory scaling table
    lines.extend(["", "## Memory Scaling", ""])
    lines.append("| N neurons | Simulator | Peak RSS (MB) | Bytes/synapse |")
    lines.append("|----------:|-----------|---------------:|--------------:|")
    for sp in results:
        bps = sp.peak_rss_mb * 1e6 / sp.n_synapses if sp.n_synapses > 0 and sp.peak_rss_mb > 0 else 0
        lines.append(
            f"| {sp.n_neurons:>9,} | {sp.simulator:<24s} "
            f"| {sp.peak_rss_mb:>13.0f} | {bps:>13.1f} |"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Scaling benchmark: SC-NeuroCore vs Brian2 vs NEST",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument(
        "--scales", type=int, nargs="+",
        default=[1000, 2000, 5000, 10000, 20000, 50000],
        help="neuron counts to benchmark (default: 1K 2K 5K 10K 20K 50K)",
    )
    ap.add_argument("--sim-ms", type=float, default=500.0, help="simulation duration in ms (default: 500)")
    ap.add_argument("--repeats", type=int, default=3, help="runs per (simulator, scale) pair (default: 3)")
    ap.add_argument(
        "--simulators", nargs="+", choices=list(SIMULATORS.keys()),
        help="simulators to run (default: all available)",
    )
    ap.add_argument("--json", type=str, help="write results to JSON file")
    ap.add_argument("--markdown", action="store_true", help="print markdown table")
    ap.add_argument("--no-gpu", action="store_true", help="skip GPU simulators")
    args = ap.parse_args()

    sims = args.simulators
    if args.no_gpu and sims is None:
        sims = [k for k in SIMULATORS if "cuda" not in k]

    sys_info = _system_info()
    params = {
        "sim_ms": args.sim_ms,
        "repeats": args.repeats,
        "conn_prob": 0.1,
        "weight_exc": 0.1,
        "g_inh": 5.0,
        "external_rate_hz": 20.0,
    }

    print("=" * 70)
    print("  SC-NeuroCore Scaling Benchmark")
    print(f"  Scales: {args.scales}")
    print(f"  Sim: {args.sim_ms} ms, Repeats: {args.repeats}")
    print(f"  System: {sys_info.get('cpu', 'unknown')}")
    if "gpu" in sys_info:
        print(f"  GPU: {sys_info['gpu']}")
    print("=" * 70)

    results = run_scaling(args.scales, args.sim_ms, args.repeats, sims)

    if args.markdown:
        md = format_markdown(results)
        print("\n" + md)

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        out = to_json(results, sys_info, params, args.scales)
        Path(args.json).write_text(json.dumps(out, indent=2, default=_serialize))
        print(f"\nResults written to {args.json}")

    if not args.json and not args.markdown:
        print("\n" + format_markdown(results))


if __name__ == "__main__":
    main()

# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Brian2 vs SC-NeuroCore — Publishable Brunel Network

"""
Brian2 vs SC-NeuroCore — Publishable Brunel Network Benchmark
==============================================================

Brunel balanced network (80/20 E/I, 10% connectivity, delta-PSC,
AI regime) at three scales (1K, 10K, 100K neurons).  Each configuration
runs 3 times; results reported as mean +/- std.

External drive follows Brunel (2000) Table 1: nu_ext = eta * nu_thr
where nu_thr = V_th / (J * C_E * tau_m).  Total Poisson lambda per
neuron per timestep = C_E * nu_ext * dt.

Compared backends:
  - Brian2         gold-standard SNN simulator
  - V1             StochasticLIF (per-neuron Python loop)
  - V3             FixedPointLIF Q8.8 (hardware-faithful)
  - V18            Numba JIT inner loop
  - V20            Vectorized NumPy (no per-neuron loop)

Usage::

    python benchmarks/brian2_benchmark.py                     # 1K only
    python benchmarks/brian2_benchmark.py --scales 1000 10000 # 1K + 10K
    python benchmarks/brian2_benchmark.py --all                # 1K, 10K, 100K
    python benchmarks/brian2_benchmark.py --repeats 5 --json results.json
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
import tracemalloc
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from brunel_translator import (
    BrunelParams,
    translate_v1_stochastic_lif,
    translate_v3_fixed_point,
    translate_v18_numba,
    translate_v20_vectorized_numpy,
)

SCALES = [1_000, 10_000, 100_000]
DEFAULT_REPEATS = 3

# Brunel (2000) AI regime: g=5, eta=2
G_INH = 5.0
ETA = 2.0  # nu_ext / nu_thr


@dataclass
class RunResult:
    wall_time_s: float
    total_spikes: int
    mean_rate_hz: float
    peak_memory_mb: float


@dataclass
class BenchmarkRow:
    scale: int
    backend: str
    wall_mean_s: float
    wall_std_s: float
    spikes_mean: float
    rate_mean_hz: float
    rate_std_hz: float
    memory_mean_mb: float
    n_runs: int
    runs: list[dict] = field(default_factory=list)


def _brunel_external_rate(bp: BrunelParams) -> float:
    """Per-connection external Poisson rate nu_ext (Hz).

    Brunel (2000): nu_thr = V_th / (J * C_E * tau_m), nu_ext = eta * nu_thr.
    """
    c_e = bp.conn_prob * bp.n_exc
    if c_e == 0:
        return 0.0
    # tau_m in seconds
    nu_thr = bp.v_threshold / (bp.weight_exc * c_e * bp.tau_mem * 1e-3)
    return ETA * nu_thr


def _ext_poisson_lambda(bp: BrunelParams) -> float:
    """Expected external spikes per neuron per timestep = C_E * nu_ext * dt_s."""
    c_e = bp.conn_prob * bp.n_exc
    return c_e * _brunel_external_rate(bp) * bp.dt / 1000.0


def _bp_for_scale(n: int) -> BrunelParams:
    n_exc = int(n * 0.8)
    n_inh = n - n_exc
    return BrunelParams(
        n_exc=n_exc,
        n_inh=n_inh,
        sim_ms=1000.0,
        g_inh=G_INH,
    )


def _measure(fn, *args, **kwargs) -> RunResult:
    """Run fn, return wall time + peak memory."""
    tracemalloc.start()
    t0 = time.perf_counter()
    spikes, rate = fn(*args, **kwargs)
    wall = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return RunResult(
        wall_time_s=wall,
        total_spikes=spikes,
        mean_rate_hz=rate,
        peak_memory_mb=peak / (1024 * 1024),
    )


# ---------------------------------------------------------------------------
# Brian2
# ---------------------------------------------------------------------------
def _run_brian2(bp: BrunelParams) -> tuple[int, float]:
    import brian2

    brian2.start_scope()

    eqs = """
    dv/dt = -v / (tau * ms) : 1
    tau : 1
    """
    G = brian2.NeuronGroup(
        bp.n_total,
        eqs,
        threshold="v > v_th",
        reset="v = v_reset",
        method="euler",
        dt=bp.dt * brian2.ms,
    )
    G.v = 0
    G.tau = bp.tau_mem
    G.namespace["v_th"] = bp.v_threshold
    G.namespace["v_reset"] = bp.v_reset

    S_exc = brian2.Synapses(G[: bp.n_exc], G, on_pre="v_post += w", dt=bp.dt * brian2.ms)
    S_exc.connect(p=bp.conn_prob)
    S_exc.namespace["w"] = bp.weight_exc

    S_inh = brian2.Synapses(G[bp.n_exc :], G, on_pre="v_post -= w", dt=bp.dt * brian2.ms)
    S_inh.connect(p=bp.conn_prob)
    S_inh.namespace["w"] = bp.weight_inh

    # Independent Poisson input per neuron (Brunel 2000 model).
    # PoissonInput avoids the correlated-drive artifact of shared PoissonGroup.
    c_ext = int(bp.conn_prob * bp.n_exc)
    nu_ext = _brunel_external_rate(bp)
    P_ext = brian2.PoissonInput(G, "v", N=c_ext, rate=nu_ext * brian2.Hz, weight=bp.weight_exc)

    mon = brian2.SpikeMonitor(G)
    brian2.run(bp.sim_ms * brian2.ms)

    rate = mon.num_spikes / (bp.sim_ms / 1000.0) / bp.n_total
    return mon.num_spikes, rate


# ---------------------------------------------------------------------------
# V1: StochasticLIF (per-neuron loop)
# ---------------------------------------------------------------------------
def _run_v1(bp: BrunelParams) -> tuple[int, float]:
    from sc_neurocore import StochasticLIFNeuron

    params = translate_v1_stochastic_lif(bp)
    rng = np.random.default_rng(bp.seed)
    neurons = [StochasticLIFNeuron(**params["neuron_kwargs"]) for _ in range(bp.n_total)]

    conn_mask = rng.random((bp.n_total, bp.n_total)) < bp.conn_prob
    np.fill_diagonal(conn_mask, False)
    weights = np.where(conn_mask, params["weight_exc"], 0.0)
    weights[bp.n_exc :, :] *= -bp.g_inh

    steps = int(bp.sim_ms / bp.dt)
    spike_count = 0
    prev_spikes = np.zeros(bp.n_total, dtype=bool)
    ext_lambda = _ext_poisson_lambda(bp)

    for _ in range(steps):
        ext_events = rng.poisson(ext_lambda, bp.n_total)
        syn_dv = weights[prev_spikes].sum(axis=0) if prev_spikes.any() else np.zeros(bp.n_total)

        spikes = np.zeros(bp.n_total, dtype=bool)
        for i, n in enumerate(neurons):
            n.v += ext_events[i] * params["ext_weight"] + syn_dv[i]
            spikes[i] = n.step(0.0)

        prev_spikes = spikes
        spike_count += int(spikes.sum())

    rate = spike_count / (bp.sim_ms / 1000.0) / bp.n_total
    return spike_count, rate


# ---------------------------------------------------------------------------
# V3: FixedPointLIF Q8.8
# ---------------------------------------------------------------------------
def _run_v3(bp: BrunelParams) -> tuple[int, float]:
    from sc_neurocore import FixedPointLIFNeuron

    params = translate_v3_fixed_point(bp)
    neurons = [
        FixedPointLIFNeuron(
            data_width=params["data_width"],
            fraction=params["fraction"],
            v_threshold=params["v_threshold_q"],
            v_reset=params["v_reset_q"],
            refractory_period=params["refractory_period"],
        )
        for _ in range(bp.n_total)
    ]

    rng = np.random.default_rng(bp.seed)
    conn_mask = rng.random((bp.n_total, bp.n_total)) < bp.conn_prob
    np.fill_diagonal(conn_mask, False)
    w_q = np.where(conn_mask, params["j_exc_q"], 0)
    w_q[bp.n_exc :, :] = np.where(conn_mask[bp.n_exc :, :], -params["j_inh_q"], 0)

    steps = int(bp.sim_ms / bp.dt)
    spike_count = 0
    prev_spikes = np.zeros(bp.n_total, dtype=bool)
    ext_lambda = _ext_poisson_lambda(bp)

    for _ in range(steps):
        ext_events = rng.poisson(ext_lambda, bp.n_total)
        syn_q = (
            w_q[prev_spikes].sum(axis=0) if prev_spikes.any() else np.zeros(bp.n_total, dtype=int)
        )

        spikes = np.zeros(bp.n_total, dtype=bool)
        for i, n in enumerate(neurons):
            I_ext = int(ext_events[i]) * params["j_exc_q"]
            I_total = int(syn_q[i]) + I_ext
            spike, _ = n.step(leak_k=params["leak_k"], gain_k=params["gain_k"], I_t=I_total)
            spikes[i] = bool(spike)

        prev_spikes = spikes
        spike_count += int(spikes.sum())

    rate = spike_count / (bp.sim_ms / 1000.0) / bp.n_total
    return spike_count, rate


# ---------------------------------------------------------------------------
# V18: Numba JIT
# ---------------------------------------------------------------------------
try:
    from numba import njit as _njit

    @_njit(cache=True)
    def _numba_brunel_loop(
        v, weights, alpha, v_rest, v_threshold, v_reset, ext_weight, ext_lambda, n, steps, seed
    ):
        np.random.seed(seed)
        spike_count = 0
        prev_spikes = np.zeros(n, dtype=np.bool_)
        for _ in range(steps):
            ext_events = np.random.poisson(ext_lambda, n)
            syn_dv = np.zeros(n)
            for j in range(n):
                if prev_spikes[j]:
                    for k in range(n):
                        syn_dv[k] += weights[j, k]
            new_spikes = np.zeros(n, dtype=np.bool_)
            for i in range(n):
                v[i] += ext_events[i] * ext_weight + syn_dv[i]
                v[i] += alpha * (v_rest - v[i])
                if v[i] >= v_threshold:
                    new_spikes[i] = True
                    v[i] = v_reset
                    spike_count += 1
            prev_spikes = new_spikes
        return spike_count

    # Trigger compilation once at import time
    _numba_brunel_loop(np.zeros(2), np.zeros((2, 2)), 0.005, 0.0, 20.0, 10.0, 0.1, 2.0, 2, 1, 0)
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False
    _numba_brunel_loop = None  # type: ignore[assignment]


def _run_v18(bp: BrunelParams) -> tuple[int, float]:
    params = translate_v18_numba(bp)
    rng = np.random.default_rng(bp.seed)

    n = bp.n_total
    v = np.full(n, bp.v_rest)
    conn_mask = rng.random((n, n)) < bp.conn_prob
    np.fill_diagonal(conn_mask, False)
    weights = np.where(conn_mask, params["weight_exc"], 0.0)
    weights[bp.n_exc :, :] *= -bp.g_inh

    alpha = bp.dt / bp.tau_mem
    steps = int(bp.sim_ms / bp.dt)
    ext_lambda = _ext_poisson_lambda(bp)

    if _HAS_NUMBA:
        spike_count = _numba_brunel_loop(
            v,
            weights,
            alpha,
            bp.v_rest,
            bp.v_threshold,
            bp.v_reset,
            params["ext_weight"],
            ext_lambda,
            n,
            steps,
            bp.seed,
        )
    else:
        spike_count = 0
        prev_spikes = np.zeros(n, dtype=bool)
        for _ in range(steps):
            ext_events = rng.poisson(ext_lambda, n)
            syn_dv = weights[prev_spikes].sum(axis=0) if prev_spikes.any() else np.zeros(n)
            v += ext_events * params["ext_weight"] + syn_dv
            v += alpha * (bp.v_rest - v)
            fired = v >= bp.v_threshold
            spike_count += int(fired.sum())
            v[fired] = bp.v_reset
            prev_spikes = fired

    rate = spike_count / (bp.sim_ms / 1000.0) / n
    return spike_count, rate


# ---------------------------------------------------------------------------
# V20: Vectorized NumPy
# ---------------------------------------------------------------------------
def _build_weight_matrix(n, n_exc, conn_prob, w_exc, w_inh, rng):
    """Build connectivity matrix. Dense for N<=10K, sparse CSR above."""
    use_sparse = n > 10_000
    if use_sparse:
        from scipy.sparse import random as sp_random

        W = sp_random(n, n, density=conn_prob, format="csr", random_state=rng)
        W.data[:] = w_exc
        W.setdiag(0)
        W.eliminate_zeros()
        for i in range(n_exc, n):
            s, e = W.indptr[i], W.indptr[i + 1]
            W.data[s:e] = -w_inh
        return W.T.tocsr(), True
    else:
        conn_mask = rng.random((n, n)) < conn_prob
        np.fill_diagonal(conn_mask, False)
        weights = np.where(conn_mask, w_exc, 0.0)
        weights[n_exc:, :] *= -(w_inh / w_exc)
        return weights, False


def _run_v20(bp: BrunelParams) -> tuple[int, float]:
    params = translate_v20_vectorized_numpy(bp)
    rng = np.random.default_rng(bp.seed)
    n = params["n_total"]
    n_exc = params["n_exc"]
    w_exc = params["weight_exc"]
    w_inh = params["weight_inh"]

    W, is_sparse = _build_weight_matrix(n, n_exc, params["conn_prob"], w_exc, w_inh, rng)

    v = np.full(n, params["v_rest"])
    alpha = params["dt"] / params["tau_mem"]
    steps = int(bp.sim_ms / params["dt"])
    spike_count = 0
    prev_spikes = np.zeros(n, dtype=bool)
    ext_lambda = _ext_poisson_lambda(bp)

    for _ in range(steps):
        ext_events = rng.poisson(ext_lambda, n)
        if is_sparse:
            I_syn = W @ prev_spikes.astype(np.float64)
        else:
            I_syn = W[prev_spikes].sum(axis=0) if prev_spikes.any() else np.zeros(n)
        v += ext_events * w_exc + I_syn
        v += alpha * (params["v_rest"] - v)
        fired = v >= params["v_threshold"]
        spike_count += int(fired.sum())
        v[fired] = params["v_reset"]
        prev_spikes = fired

    rate = spike_count / (bp.sim_ms / 1000.0) / n
    return spike_count, rate


# ---------------------------------------------------------------------------
# Backend registry
# ---------------------------------------------------------------------------
BACKENDS: dict[str, tuple[str, callable]] = {
    "brian2": ("Brian2 (reference)", _run_brian2),
    "v1": ("SC V1 StochasticLIF", _run_v1),
    "v3": ("SC V3 FixedPoint Q8.8", _run_v3),
    "v18": ("SC V18 Numba JIT", _run_v18),
    "v20": ("SC V20 Vectorized", _run_v20),
}


def _run_backend(key: str, bp: BrunelParams, repeats: int) -> BenchmarkRow | None:
    label, fn = BACKENDS[key]

    if key == "brian2":
        try:
            import brian2  # noqa: F401
        except ImportError:
            print("  [SKIP] Brian2 not installed — pip install brian2", file=sys.stderr)
            return None

    # Per-neuron loop backends are too slow above 10K
    if key in ("v1", "v3") and bp.n_total > 10_000:
        print(
            f"  [SKIP] {label} — per-neuron loop infeasible at {bp.n_total} neurons",
            file=sys.stderr,
        )
        return None

    # V18 uses dense N×N matrix — prohibitive above 20K
    if key == "v18" and bp.n_total > 20_000:
        print(
            f"  [SKIP] {label} — dense N×N matrix ({bp.n_total**2 * 8 / 1e9:.1f} GB) "
            f"at {bp.n_total} neurons",
            file=sys.stderr,
        )
        return None

    runs: list[RunResult] = []
    for r in range(repeats):
        print(f"  {label} run {r + 1}/{repeats} ...", end=" ", flush=True)
        try:
            result = _measure(fn, bp)
            runs.append(result)
            print(
                f"{result.wall_time_s:.2f}s, {result.total_spikes} spikes, "
                f"{result.mean_rate_hz:.1f} Hz"
            )
        except Exception as exc:
            print(f"FAILED: {exc}", file=sys.stderr)
            return None

    walls = np.array([r.wall_time_s for r in runs])
    spikes = np.array([r.total_spikes for r in runs])
    rates = np.array([r.mean_rate_hz for r in runs])
    mems = np.array([r.peak_memory_mb for r in runs])

    return BenchmarkRow(
        scale=bp.n_total,
        backend=label,
        wall_mean_s=float(walls.mean()),
        wall_std_s=float(walls.std()),
        spikes_mean=float(spikes.mean()),
        rate_mean_hz=float(rates.mean()),
        rate_std_hz=float(rates.std()),
        memory_mean_mb=float(mems.mean()),
        n_runs=repeats,
        runs=[asdict(r) for r in runs],
    )


# ---------------------------------------------------------------------------
# Output formatters
# ---------------------------------------------------------------------------
def _system_info() -> dict:
    info = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu": platform.processor() or "unknown",
        "numpy": np.__version__,
    }
    try:
        import brian2

        info["brian2"] = brian2.__version__
    except ImportError:
        pass
    try:
        import numba

        info["numba"] = numba.__version__
    except ImportError:
        pass
    return info


def _print_markdown(rows: list[BenchmarkRow]) -> None:
    print()
    print("## Brian2 vs SC-NeuroCore — Brunel Balanced Network (AI Regime)")
    print()
    print("| Scale | Backend | Wall (s) | Spikes | Rate (Hz) | Memory (MB) |")
    print("|------:|:--------|--------:|---------:|----------:|------------:|")
    for r in rows:
        print(
            f"| {r.scale:,} | {r.backend} | "
            f"{r.wall_mean_s:.3f} +/- {r.wall_std_s:.3f} | "
            f"{r.spikes_mean:,.0f} | "
            f"{r.rate_mean_hz:.1f} +/- {r.rate_std_hz:.1f} | "
            f"{r.memory_mean_mb:.1f} |"
        )
    print()

    brian2_by_scale: dict[int, float] = {}
    for r in rows:
        if "Brian2" in r.backend:
            brian2_by_scale[r.scale] = r.wall_mean_s

    if brian2_by_scale:
        print("### Speedup vs Brian2")
        print()
        print("| Scale | Backend | Speedup |")
        print("|------:|:--------|--------:|")
        for r in rows:
            if "Brian2" in r.backend:
                continue
            ref = brian2_by_scale.get(r.scale)
            if ref and r.wall_mean_s > 0:
                speedup = ref / r.wall_mean_s
                print(f"| {r.scale:,} | {r.backend} | {speedup:.2f}x |")
        print()


def _serialize(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    raise TypeError(f"{type(obj).__name__} not JSON serializable")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Brian2 vs SC-NeuroCore Brunel balanced network benchmark",
    )
    parser.add_argument(
        "--scales", nargs="+", type=int, default=[1000], help="Neuron counts (default: 1000)"
    )
    parser.add_argument("--all", action="store_true", help="Run 1K, 10K, 100K")
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--json", type=str, default=None, help="Write JSON results to file")
    parser.add_argument(
        "--backends",
        nargs="+",
        default=list(BACKENDS.keys()),
        choices=list(BACKENDS.keys()),
        help="Backends to run (default: all)",
    )
    args = parser.parse_args()

    scales = SCALES if args.all else args.scales

    # Show Brunel parameters for reproducibility
    sample_bp = _bp_for_scale(scales[0])
    nu_ext = _brunel_external_rate(sample_bp)
    ext_lam = _ext_poisson_lambda(sample_bp)
    c_e = sample_bp.conn_prob * sample_bp.n_exc

    print(f"Brunel balanced network benchmark — {args.repeats} repeats per config")
    print(f"  Regime: AI (g={G_INH}, eta={ETA})")
    print(
        f"  V_th={sample_bp.v_threshold} mV, V_reset={sample_bp.v_reset} mV, "
        f"tau_m={sample_bp.tau_mem} ms, J={sample_bp.weight_exc} mV"
    )
    print(f"  C_E={c_e:.0f}, nu_ext={nu_ext:.1f} Hz, " f"ext_lambda/step={ext_lam:.4f}")
    print(f"  Scales: {scales}")
    print(f"  Backends: {args.backends}")
    print()

    all_rows: list[BenchmarkRow] = []

    for n in scales:
        bp = _bp_for_scale(n)
        c_e_n = bp.conn_prob * bp.n_exc
        nu_ext_n = _brunel_external_rate(bp)
        print(
            f"=== {n:,} neurons ({bp.n_exc}E / {bp.n_inh}I), "
            f"C_E={c_e_n:.0f}, nu_ext={nu_ext_n:.1f} Hz, 1s sim ==="
        )

        for key in args.backends:
            row = _run_backend(key, bp, args.repeats)
            if row is not None:
                all_rows.append(row)
        print()

    _print_markdown(all_rows)

    if args.json:
        out = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "system": _system_info(),
            "brunel_params": {
                "regime": "AI",
                "g_inh": G_INH,
                "eta": ETA,
                "sim_ms": 1000.0,
                "dt": 0.1,
                "conn_prob": 0.1,
                "weight_exc": 0.1,
                "v_threshold": 20.0,
                "v_reset": 10.0,
                "tau_mem": 20.0,
                "repeats": args.repeats,
                "scales": scales,
            },
            "rows": [asdict(r) for r in all_rows],
        }
        path = Path(args.json)
        path.write_text(json.dumps(out, indent=2, default=_serialize))
        print(f"Results written to {path}")


if __name__ == "__main__":
    main()

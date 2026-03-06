# SPDX-License-Identifier: AGPL-3.0-or-later
"""
SNN Simulator Comparison Benchmark
====================================

Head-to-head wall-clock comparison of SC-NeuroCore vs. NEST, Brian2, and Lava
on standard SNN workloads.

Usage::

    python benchmarks/snn_comparison.py              # SC-NeuroCore only
    python benchmarks/snn_comparison.py --all         # all available backends
    python benchmarks/snn_comparison.py --markdown    # markdown output

Requires optional deps::

    pip install nest-simulator brian2 lava-nc
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass
class BenchResult:
    backend: str
    workload: str
    n_neurons: int
    n_synapses: int
    sim_time_ms: float
    wall_clock_s: float
    spikes_total: int
    throughput_kevents_s: float = 0.0
    notes: str = ""

    def __post_init__(self) -> None:
        if self.wall_clock_s > 0 and self.spikes_total > 0:
            self.throughput_kevents_s = (self.spikes_total / self.wall_clock_s) / 1000


# ---------------------------------------------------------------------------
# Brunel balanced network parameters
# ---------------------------------------------------------------------------
BRUNEL_DEFAULTS = dict(
    n_exc=800,
    n_inh=200,
    conn_prob=0.1,
    weight_exc=0.1,  # mV
    g_inh=5.0,
    sim_ms=1000.0,
    dt=0.1,
    v_threshold=20.0,  # mV
    v_reset=10.0,
    tau_mem=20.0,  # ms
    external_rate_hz=20.0,
)


# ---------------------------------------------------------------------------
# SC-NeuroCore backend
# ---------------------------------------------------------------------------
def bench_scneurocore_brunel(**kw) -> BenchResult:
    from sc_neurocore import StochasticLIFNeuron

    p = {**BRUNEL_DEFAULTS, **kw}
    n_total = p["n_exc"] + p["n_inh"]

    rng = np.random.default_rng(42)
    neurons = [
        StochasticLIFNeuron(
            v_threshold=p["v_threshold"],
            tau_mem=p["tau_mem"],
            dt=p["dt"],
        )
        for _ in range(n_total)
    ]

    # Build sparse connectivity
    conn_mask = rng.random((n_total, n_total)) < p["conn_prob"]
    np.fill_diagonal(conn_mask, False)
    weights = np.where(conn_mask, p["weight_exc"], 0.0)
    weights[p["n_exc"] :, :] *= -p["g_inh"]
    n_synapses = int(conn_mask.sum())

    steps = int(p["sim_ms"] / p["dt"])
    spike_count = 0
    prev_spikes = np.zeros(n_total, dtype=bool)

    t0 = time.perf_counter()
    for _ in range(steps):
        ext_current = (
            rng.poisson(p["external_rate_hz"] * p["dt"] / 1000.0, n_total) * p["weight_exc"]
        )
        syn_input = weights[prev_spikes].sum(axis=0) if prev_spikes.any() else np.zeros(n_total)
        total_input = ext_current + syn_input
        spikes = np.array([n.step(float(total_input[i])) for i, n in enumerate(neurons)])
        prev_spikes = spikes.astype(bool)
        spike_count += int(spikes.sum())
    wall = time.perf_counter() - t0

    return BenchResult(
        backend="sc-neurocore",
        workload="brunel_balanced",
        n_neurons=n_total,
        n_synapses=n_synapses,
        sim_time_ms=p["sim_ms"],
        wall_clock_s=wall,
        spikes_total=spike_count,
    )


# ---------------------------------------------------------------------------
# NEST backend
# ---------------------------------------------------------------------------
def bench_nest_brunel(**kw) -> BenchResult | None:
    try:
        import nest
    except ImportError:
        return None

    p = {**BRUNEL_DEFAULTS, **kw}
    n_total = p["n_exc"] + p["n_inh"]

    nest.ResetKernel()
    nest.set(resolution=p["dt"], rng_seed=42)

    neurons = nest.Create(
        "iaf_psc_delta",
        n_total,
        params={
            "V_th": p["v_threshold"],
            "V_reset": p["v_reset"],
            "tau_m": p["tau_mem"],
        },
    )
    exc = neurons[: p["n_exc"]]
    inh = neurons[p["n_exc"] :]

    noise = nest.Create("poisson_generator", params={"rate": p["external_rate_hz"]})
    nest.Connect(noise, neurons, syn_spec={"weight": p["weight_exc"]})

    nest.Connect(
        exc,
        neurons,
        conn_spec={"rule": "pairwise_bernoulli", "p": p["conn_prob"]},
        syn_spec={"weight": p["weight_exc"]},
    )
    nest.Connect(
        inh,
        neurons,
        conn_spec={"rule": "pairwise_bernoulli", "p": p["conn_prob"]},
        syn_spec={"weight": -p["g_inh"] * p["weight_exc"]},
    )

    sr = nest.Create("spike_recorder")
    nest.Connect(neurons, sr)

    n_synapses = int(nest.GetKernelStatus("num_connections"))

    t0 = time.perf_counter()
    nest.Simulate(p["sim_ms"])
    wall = time.perf_counter() - t0

    events = nest.GetStatus(sr, "events")[0]
    spike_count = len(events["times"])

    return BenchResult(
        backend="nest",
        workload="brunel_balanced",
        n_neurons=n_total,
        n_synapses=n_synapses,
        sim_time_ms=p["sim_ms"],
        wall_clock_s=wall,
        spikes_total=spike_count,
    )


# ---------------------------------------------------------------------------
# Brian2 backend
# ---------------------------------------------------------------------------
def bench_brian2_brunel(**kw) -> BenchResult | None:
    try:
        import brian2
    except ImportError:
        return None

    p = {**BRUNEL_DEFAULTS, **kw}
    n_total = p["n_exc"] + p["n_inh"]

    brian2.start_scope()

    eqs = """
    dv/dt = (-v + I_ext) / (tau * ms) : 1
    I_ext : 1
    tau : 1
    """

    G = brian2.NeuronGroup(
        n_total,
        eqs,
        threshold="v > v_th",
        reset="v = v_reset",
        method="euler",
        dt=p["dt"] * brian2.ms,
    )
    G.v = 0
    G.tau = p["tau_mem"]
    G.namespace["v_th"] = p["v_threshold"]
    G.namespace["v_reset"] = p["v_reset"]

    exc_group = G[: p["n_exc"]]
    inh_group = G[p["n_exc"] :]

    S_exc = brian2.Synapses(exc_group, G, on_pre="v_post += w", dt=p["dt"] * brian2.ms)
    S_exc.connect(p=p["conn_prob"])
    S_exc.namespace["w"] = p["weight_exc"]

    S_inh = brian2.Synapses(inh_group, G, on_pre="v_post -= w", dt=p["dt"] * brian2.ms)
    S_inh.connect(p=p["conn_prob"])
    S_inh.namespace["w"] = p["g_inh"] * p["weight_exc"]

    poisson = brian2.PoissonInput(
        G, "I_ext", 1, p["external_rate_hz"] * brian2.Hz, weight=p["weight_exc"]
    )

    mon = brian2.SpikeMonitor(G)

    n_synapses = len(S_exc) + len(S_inh)

    t0 = time.perf_counter()
    brian2.run(p["sim_ms"] * brian2.ms)
    wall = time.perf_counter() - t0

    return BenchResult(
        backend="brian2",
        workload="brunel_balanced",
        n_neurons=n_total,
        n_synapses=n_synapses,
        sim_time_ms=p["sim_ms"],
        wall_clock_s=wall,
        spikes_total=mon.num_spikes,
    )


# ---------------------------------------------------------------------------
# Lava backend (stub — Lava API is substantially different)
# ---------------------------------------------------------------------------
def bench_lava_brunel(**kw) -> BenchResult | None:
    try:
        from lava.proc.lif.process import LIF
        from lava.proc.dense.process import Dense
        from lava.magma.core.run_configs import Loihi2SimCfg
        from lava.magma.core.run_conditions import RunSteps
    except ImportError:
        return None

    p = {**BRUNEL_DEFAULTS, **kw}
    n_total = p["n_exc"] + p["n_inh"]

    rng = np.random.default_rng(42)
    conn = (rng.random((n_total, n_total)) < p["conn_prob"]).astype(np.int32)
    np.fill_diagonal(conn, 0)
    weights = conn * int(p["weight_exc"] * 256)
    weights[p["n_exc"] :, :] *= -int(p["g_inh"])

    lif = LIF(
        shape=(n_total,),
        vth=int(p["v_threshold"] * 256),
        du=int(256 * p["dt"] / p["tau_mem"]),
        dv=0,
    )
    dense = Dense(weights=weights)
    dense.s_out.connect(lif.a_in)
    lif.s_out.connect(dense.s_in)

    steps = int(p["sim_ms"] / p["dt"])

    t0 = time.perf_counter()
    lif.run(condition=RunSteps(num_steps=steps), run_cfg=Loihi2SimCfg())
    wall = time.perf_counter() - t0
    lif.stop()

    return BenchResult(
        backend="lava",
        workload="brunel_balanced",
        n_neurons=n_total,
        n_synapses=int(conn.sum()),
        sim_time_ms=p["sim_ms"],
        wall_clock_s=wall,
        spikes_total=0,
        notes="spike count requires Loihi 2 hardware probe API",
    )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
ALL_BACKENDS = [
    ("sc-neurocore", bench_scneurocore_brunel),
    ("nest", bench_nest_brunel),
    ("brian2", bench_brian2_brunel),
    ("lava", bench_lava_brunel),
]


def run_comparison(backends: list[str] | None = None, **kw) -> list[BenchResult]:
    results: list[BenchResult] = []
    for name, fn in ALL_BACKENDS:
        if backends and name not in backends:
            continue
        print(f"  Running {name}...", end=" ", flush=True)
        r = fn(**kw)
        if r is None:
            print("SKIPPED (not installed)")
        else:
            print(f"{r.wall_clock_s:.3f}s, {r.spikes_total} spikes")
            results.append(r)
    return results


def format_markdown(results: list[BenchResult]) -> str:
    lines = [
        "| Backend | Neurons | Synapses | Sim (ms) | Wall (s) | Spikes | Throughput (kevt/s) |",
        "|---------|--------:|---------:|---------:|---------:|-------:|--------------------:|",
    ]
    for r in results:
        lines.append(
            f"| {r.backend} | {r.n_neurons} | {r.n_synapses:,} | {r.sim_time_ms:.0f} "
            f"| {r.wall_clock_s:.3f} | {r.spikes_total:,} | {r.throughput_kevents_s:.1f} |"
        )
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="SNN simulator comparison benchmark")
    ap.add_argument("--all", action="store_true", help="run all available backends")
    ap.add_argument("--markdown", action="store_true", help="output markdown table")
    ap.add_argument("--json", type=str, help="write results to JSON file")
    ap.add_argument("--sim-ms", type=float, default=1000.0)
    args = ap.parse_args()

    backends = None if args.all else ["sc-neurocore"]

    print("Brunel Balanced Network Benchmark")
    print("=" * 40)
    results = run_comparison(backends=backends, sim_ms=args.sim_ms)

    if args.markdown:
        print("\n" + format_markdown(results))

    if args.json:
        Path(args.json).write_text(json.dumps([asdict(r) for r in results], indent=2))
        print(f"\nResults written to {args.json}")


if __name__ == "__main__":
    main()

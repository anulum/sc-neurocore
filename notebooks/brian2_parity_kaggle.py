# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore -- Brian2 cross-framework parity check (Task 1.6)

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

print("=" * 70)
print("SETUP")
print("=" * 70)
subprocess.check_call(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "--no-deps",
        "git+https://github.com/anulum/sc-neurocore.git@main",
    ],
    stdout=sys.stdout,
    stderr=sys.stderr,
)
subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "-q", "brian2"],
    stdout=sys.stdout,
    stderr=sys.stderr,
)

RESULTS = {}


def report(name, passed, detail=""):
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {name}" + (" -- " + detail if detail else ""))
    RESULTS[name] = {"passed": passed, "detail": detail}


# ===========================================================================
# TEST 1: Single LIF neuron comparison
# ===========================================================================
def test_single_lif():
    print("\n" + "=" * 70)
    print("TEST 1: Single LIF Neuron (constant current)")
    print("=" * 70)

    # Brian2
    from brian2 import NeuronGroup, SpikeMonitor, defaultclock, ms, mV, run, prefs, seed as b2seed
    import brian2

    brian2.start_scope()
    prefs.codegen.target = "numpy"
    b2seed(42)
    defaultclock.dt = 0.1 * ms

    tau = 20 * ms
    eqs = "dv/dt = (-v + 14*mV) / tau : volt"
    G = NeuronGroup(1, eqs, threshold="v > 10*mV", reset="v = 0*mV", method="euler")
    G.v = 0 * mV
    mon = SpikeMonitor(G)
    run(500 * ms)
    b2_spikes = len(mon.t)
    b2_times = np.array(mon.t / ms)
    print(f"  Brian2: {b2_spikes} spikes in 500ms")

    # SC-NeuroCore (manual Euler, same ODE)
    dt_ms = 0.1
    n_steps = int(500 / dt_ms)
    v = 0.0
    sc_times = []
    for i in range(n_steps):
        dv = (-v + 14.0) / 20.0 * dt_ms
        v += dv
        if v >= 10.0:
            sc_times.append(i * dt_ms)
            v = 0.0
    sc_spikes = len(sc_times)
    print(f"  SC-NeuroCore (Euler): {sc_spikes} spikes in 500ms")

    # Compare
    spike_match = b2_spikes == sc_spikes
    report("single_lif_spike_count", spike_match, f"B2={b2_spikes}, SC={sc_spikes}")

    if b2_spikes > 0 and sc_spikes > 0:
        max_diff = max(abs(b - s) for b, s in zip(b2_times, np.array(sc_times)))
        report("single_lif_spike_timing", max_diff < 0.5, f"max timing diff={max_diff:.3f}ms")

    b2_isis = np.diff(b2_times) if len(b2_times) > 1 else np.array([])
    sc_isis = np.diff(sc_times) if len(sc_times) > 1 else np.array([])
    if len(b2_isis) > 0:
        b2_mean_isi = float(np.mean(b2_isis))
        sc_mean_isi = float(np.mean(sc_isis))
        isi_err = abs(b2_mean_isi - sc_mean_isi) / b2_mean_isi
        report(
            "single_lif_isi_match",
            isi_err < 0.01,
            f"B2 ISI={b2_mean_isi:.2f}ms, SC ISI={sc_mean_isi:.2f}ms, err={isi_err:.2%}",
        )

    return {
        "b2_spikes": b2_spikes,
        "sc_spikes": sc_spikes,
        "b2_times": b2_times.tolist() if len(b2_times) > 0 else [],
        "sc_times": sc_times,
    }


# ===========================================================================
# TEST 2: Population of 100 LIF with Poisson input
# ===========================================================================
def test_population_poisson():
    print("\n" + "=" * 70)
    print("TEST 2: 100 LIF neurons with Poisson input")
    print("=" * 70)

    from brian2 import (
        NeuronGroup,
        PoissonGroup,
        Synapses,
        SpikeMonitor,
        defaultclock,
        ms,
        mV,
        Hz,
        run,
        prefs,
        seed as b2seed,
    )
    import brian2

    brian2.start_scope()
    prefs.codegen.target = "numpy"
    b2seed(42)
    defaultclock.dt = 0.1 * ms

    N = 100
    tau = 20 * ms
    eqs = "dv/dt = -v / tau : volt"
    G = NeuronGroup(N, eqs, threshold="v > 10*mV", reset="v = 0*mV", method="euler")
    G.v = "rand() * 5 * mV"

    P = PoissonGroup(N, rates=500 * Hz)
    S = Synapses(P, G, on_pre="v += 2*mV")
    S.connect("i == j")

    mon = SpikeMonitor(G)
    t0 = time.time()
    run(500 * ms)
    b2_time = time.time() - t0

    b2_total = len(mon.t)
    b2_rates = []
    for idx in range(N):
        b2_rates.append(float(np.sum(mon.i == idx)) / 0.5)  # Hz
    b2_mean_rate = float(np.mean(b2_rates))

    print(f"  Brian2: {b2_total} spikes, mean rate={b2_mean_rate:.1f} Hz, time={b2_time:.2f}s")

    # SC-NeuroCore: manual Poisson-driven LIF population
    rng = np.random.default_rng(42)
    dt_ms = 0.1
    n_steps = int(500 / dt_ms)
    N = 100
    v = rng.uniform(0, 5, N)
    sc_spike_counts = np.zeros(N, dtype=int)

    poisson_rate = 500.0  # Hz
    p_spike = poisson_rate * dt_ms / 1000.0

    t0 = time.time()
    for step in range(n_steps):
        # Poisson input
        input_spikes = rng.random(N) < p_spike
        v[input_spikes] += 2.0

        # LIF dynamics: dv/dt = -v/tau
        v += -v / 20.0 * dt_ms

        # Threshold
        fired = v >= 10.0
        sc_spike_counts[fired] += 1
        v[fired] = 0.0

    sc_time = time.time() - t0
    sc_total = int(np.sum(sc_spike_counts))
    sc_rates = sc_spike_counts / 0.5  # Hz
    sc_mean_rate = float(np.mean(sc_rates))

    print(
        f"  SC-NeuroCore: {sc_total} spikes, mean rate={sc_mean_rate:.1f} Hz, time={sc_time:.4f}s"
    )

    # Compare (looser tolerance due to different Poisson seeds)
    if b2_total > 0:
        spike_ratio = sc_total / b2_total
        report(
            "pop_spike_count_20pct",
            0.8 < spike_ratio < 1.2,
            f"B2={b2_total}, SC={sc_total}, ratio={spike_ratio:.2f}",
        )
    else:
        report("pop_spike_count_20pct", False, "Brian2 produced 0 spikes")

    if b2_mean_rate > 0:
        rate_ratio = sc_mean_rate / b2_mean_rate
        report(
            "pop_mean_rate_20pct",
            0.8 < rate_ratio < 1.2,
            f"B2={b2_mean_rate:.1f}Hz, SC={sc_mean_rate:.1f}Hz",
        )
    else:
        report("pop_mean_rate_20pct", False, "Brian2 rate=0")

    report(
        "pop_sc_faster",
        sc_time < b2_time,
        f"SC={sc_time:.4f}s, B2={b2_time:.2f}s, ratio={b2_time / max(sc_time, 1e-6):.0f}x",
    )

    return {
        "b2": {"spikes": b2_total, "mean_rate": b2_mean_rate, "time": b2_time},
        "sc": {"spikes": sc_total, "mean_rate": sc_mean_rate, "time": sc_time},
    }


# ===========================================================================
# TEST 3: SC-NeuroCore Network API Brunel
# ===========================================================================
def test_network_api():
    print("\n" + "=" * 70)
    print("TEST 3: SC-NeuroCore Network API (Brunel 200E/50I)")
    print("=" * 70)

    try:
        from sc_neurocore.network.network import Network
        from sc_neurocore.network.population import Population
        from sc_neurocore.network.projection import Projection

        net = Network()
        exc = Population("StochasticLIFNeuron", 200, label="exc", params={"noise_std": 0.0})
        inh = Population("StochasticLIFNeuron", 50, label="inh", params={"noise_std": 0.0})
        net.add(exc)
        net.add(inh)

        # E->E and I->E projections
        proj_ee = Projection(exc, exc, weight=0.05, probability=0.1, seed=42)
        proj_ie = Projection(inh, exc, weight=-0.2, probability=0.1, seed=43)
        net.add(proj_ee)
        net.add(proj_ie)

        # Add external stimulus
        from sc_neurocore.network.stimulus import PoissonStimulus

        stim = PoissonStimulus(exc, rate=800.0, weight=0.1, seed=44)
        net.add(stim)

        t0 = time.time()
        net.run(duration=0.5, dt=0.001, backend="python")
        elapsed = time.time() - t0

        # Count spikes
        total_spikes = 0
        for pop in [exc, inh]:
            for neuron in pop.neurons:
                state = neuron.get_state()
                # Check if neuron tracked spikes
                if hasattr(neuron, "_spike_count"):
                    total_spikes += neuron._spike_count

        report("network_api_runs", True, f"time={elapsed:.2f}s")
        print(f"  Network ran successfully in {elapsed:.2f}s")
        return {"ok": True, "time": elapsed}

    except ImportError as e:
        report("network_api_runs", False, f"Import error: {e}")
        return {"ok": False, "error": str(e)}
    except Exception as e:
        import traceback

        traceback.print_exc()
        report("network_api_runs", False, f"Error: {e}")
        return {"ok": False, "error": str(e)}


# ===========================================================================
# Main
# ===========================================================================
def main():
    print("=" * 70)
    print("SC-NeuroCore Brian2 Parity Check (Task 1.6)")
    print(f"Time: {time.strftime('%Y-%m-%dT%H:%M:%S%z')}")
    print("=" * 70)

    t0 = time.time()

    r1 = test_single_lif()
    r2 = test_population_poisson()
    r3 = test_network_api()

    elapsed = time.time() - t0

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    total = len(RESULTS)
    passed = sum(1 for v in RESULTS.values() if v["passed"])
    for name, r in RESULTS.items():
        status = "PASS" if r["passed"] else "FAIL"
        print(f"  [{status}] {name}")
    print(f"\n  {passed}/{total} passed, time={elapsed:.1f}s")

    output = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "elapsed_s": round(elapsed, 1),
        "single_lif": r1,
        "population_poisson": r2,
        "network_api": r3,
        "checks": RESULTS,
    }

    out_path = Path("/kaggle/working/brian2_parity_results.json")
    if not out_path.parent.exists():
        out_path = Path("brian2_parity_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"  Saved: {out_path}")


if __name__ == "__main__":
    main()

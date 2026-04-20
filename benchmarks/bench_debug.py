# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — debug subsystem microbenchmark harness

"""Measures the hot paths of the offline tracer + HIL scope subsystems.

Emits a markdown table to stdout and a JSON snapshot to
benchmarks/results/bench_debug.json for doc consumption.
"""

import json
import os
import sys
import time

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "..", "src"))

from sc_neurocore.debug.analyzer import (  # noqa: E402
    causal_chain,
    find_divergence,
    spike_diff,
)
from sc_neurocore.debug.sc_doctor import ScDoctor  # noqa: E402
from sc_neurocore.debug.sc_scope import (  # noqa: E402
    BitstreamSample,
    LayerErrorBudget,
    LiveAnalyzer,
    TransportBackend,
    TransportConfig,
    TransportType,
    TriggerCondition,
    TriggerEngine,
    TriggerType,
    compute_scc,
)
from sc_neurocore.debug.tracer import ExecutionTrace  # noqa: E402


def _ns_per_call(fn, iters: int) -> float:
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    t1 = time.perf_counter()
    return (t1 - t0) * 1e9 / iters


def main() -> int:
    rng = np.random.default_rng(7)
    N, T = 32, 1000
    spikes_a = (rng.random((T, N)) < 0.05).astype(np.int8)
    spikes_b = spikes_a.copy()
    spikes_b[500, 7] = 1 - spikes_b[500, 7]
    v = rng.normal(size=(T, N))
    c = rng.normal(size=(T, N))
    trace_a = ExecutionTrace(N, T, spikes_a, v, c)
    trace_b = ExecutionTrace(N, T, spikes_b, v, c)

    results: dict[str, dict[str, float]] = {}
    results["find_divergence_T1000_N32"] = {
        "ns_per_call": _ns_per_call(lambda: find_divergence(trace_a, trace_b), 1_000),
    }
    results["spike_diff_T1000_N32"] = {
        "ns_per_call": _ns_per_call(lambda: spike_diff(trace_a, trace_b), 1_000),
    }
    results["causal_chain_depth10"] = {
        "ns_per_call": _ns_per_call(lambda: causal_chain(trace_a, 0, 999, max_depth=10), 1_000),
    }

    doc = ScDoctor(initial_length=512)
    results["scdoctor_adapt"] = {
        "ns_per_call": _ns_per_call(lambda: doc.adapt(0.10), 100_000),
    }
    doc.error_correction_enabled = True
    results["scdoctor_encode_ecc"] = {
        "ns_per_call": _ns_per_call(lambda: doc.encode_ecc(0xA), 100_000),
    }
    results["scdoctor_decode_ecc"] = {
        "ns_per_call": _ns_per_call(lambda: doc.decode_ecc(0x55), 100_000),
    }

    cfg = TransportConfig(transport_type=TransportType.SIMULATED)
    tr = TransportBackend(cfg)
    tr.connect()
    an = LiveAnalyzer(num_layers=1, window_size=1024)
    for _ in range(64):
        words = tr.read_bitstream(num_words=32, layer_id=0)
        an.ingest(
            BitstreamSample(
                timestamp_ns=time.perf_counter_ns(),
                layer_id=0,
                neuron_id=0,
                words=words,
            )
        )
    results["liveanalyzer_layer_stats"] = {
        "ns_per_call": _ns_per_call(lambda: an.layer_stats(0), 5_000),
    }

    a = rng.integers(0, 2**32, size=256, dtype=np.uint32)
    b = rng.integers(0, 2**32, size=256, dtype=np.uint32)
    results["compute_scc_256w"] = {
        "ns_per_call": _ns_per_call(lambda: compute_scc(a, b), 5_000),
    }

    eng = TriggerEngine()
    eng.add_trigger(TriggerCondition(TriggerType.DENSITY_ABOVE, threshold=0.3, layer_id=0))
    eng.add_trigger(TriggerCondition(TriggerType.DENSITY_BELOW, threshold=0.1, layer_id=0))
    sample = BitstreamSample(
        timestamp_ns=123,
        layer_id=0,
        neuron_id=0,
        words=rng.integers(0, 2**32, size=32, dtype=np.uint32),
    )

    def _trigger_evaluate():
        eng.evaluate(sample)
        if len(eng.events) > 1000:
            eng.events.clear()

    results["trigger_engine_evaluate_2conds"] = {
        "ns_per_call": _ns_per_call(_trigger_evaluate, 100_000),
    }

    budget = LayerErrorBudget(layer_id=0, expected_density=0.3, tolerance=0.05)
    counter = {"i": 0}

    def _budget_check():
        counter["i"] += 1
        budget.check(0.3 + 0.01 * (counter["i"] % 10))

    results["layer_error_budget_check"] = {
        "ns_per_call": _ns_per_call(_budget_check, 100_000),
    }

    print(f"\n{'Operation':<38} {'ns/call':>14} {'ops/s':>14}")
    print("-" * 70)
    for op, m in results.items():
        ns = m["ns_per_call"]
        ops = 1e9 / ns
        print(f"{op:<38} {ns:>14.1f} {ops:>14.0f}")

    out_dir = os.path.join(SCRIPT_DIR, "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "bench_debug.json")
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nResults written to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — edge subsystem microbenchmark harness

"""Measures the hot paths of the pure-Python edge runtime.

Emits stdout markdown + benchmarks/results/bench_edge.json.
"""

import json
import os
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "..", "src"))

from sc_neurocore.edge.bitstream import (  # noqa: E402
    popcount32,
    popcount_slice,
    scc,
)
from sc_neurocore.edge.lfsr import Lfsr16  # noqa: E402
from sc_neurocore.edge.neuron import IzhikevichNeuron, LifNeuron  # noqa: E402
from sc_neurocore.edge.sc_network import SCLayer, SCNetwork  # noqa: E402
from sc_neurocore.edge.sobol import SobolGenerator  # noqa: E402
from sc_neurocore.edge.weights import (  # noqa: E402
    deserialize_weights,
    serialize_weights,
)


def _ns_per_call(fn, iters: int) -> float:
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    t1 = time.perf_counter()
    return (t1 - t0) * 1e9 / iters


def main() -> int:
    results: dict[str, dict[str, float]] = {}

    results["popcount32"] = {
        "ns_per_call": _ns_per_call(lambda: popcount32(0xDEADBEEF), 1_000_000),
    }

    words_1024 = [0xDEADBEEF] * 1024
    results["popcount_slice_1024w"] = {
        "ns_per_call": _ns_per_call(lambda: popcount_slice(words_1024), 10_000),
    }

    lfsr = Lfsr16(0xACE1)
    results["lfsr16_encode_1024bit"] = {
        "ns_per_call": _ns_per_call(lambda: lfsr.encode(32768, 1024), 5_000),
    }

    sob = SobolGenerator()
    results["sobol_encode_1024bit"] = {
        "ns_per_call": _ns_per_call(lambda: sob.encode(32768, 1024), 5_000),
    }

    lif = LifNeuron(threshold=512, leak_shift=3)
    lif_input = [0xAAAAAAAA] * 32
    results["lif_neuron_tick_32w"] = {
        "ns_per_call": _ns_per_call(lambda: lif.tick(lif_input), 1_000_000),
    }

    izh = IzhikevichNeuron.regular_spiking()
    I_q16 = 10 << 16
    results["izhikevich_tick"] = {
        "ns_per_call": _ns_per_call(lambda: izh.tick(I_q16), 1_000_000),
    }

    net = SCNetwork(bit_length=1024)
    net.add_layer(SCLayer(n_inputs=32, n_outputs=16))
    net.add_layer(SCLayer(n_inputs=16, n_outputs=8))
    probs = [0.5] * 32
    results["sc_network_run_32_16_8"] = {
        "ns_per_call": _ns_per_call(lambda: net.run(probs), 50),
    }

    layers_data = net.export_weights()
    blob = serialize_weights(layers_data)
    results["serialize_weights_2layer"] = {
        "ns_per_call": _ns_per_call(lambda: serialize_weights(layers_data), 1_000),
        "blob_bytes": float(len(blob)),
    }
    results["deserialize_weights_2layer"] = {
        "ns_per_call": _ns_per_call(lambda: deserialize_weights(blob), 1_000),
    }

    a = [0xDEADBEEF] * 32
    b = [0xCAFEBABE] * 32
    results["scc_32w"] = {
        "ns_per_call": _ns_per_call(lambda: scc(a, b, 32 * 32), 5_000),
    }

    print(f"\n{'Operation':<36} {'ns/call':>14} {'ops/s':>14}")
    print("-" * 68)
    for op, m in results.items():
        ns = m["ns_per_call"]
        ops = 1e9 / ns
        print(f"{op:<36} {ns:>14.1f} {ops:>14.0f}")

    out_dir = os.path.join(SCRIPT_DIR, "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "bench_edge.json")
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nResults written to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

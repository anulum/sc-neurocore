# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — proto schema microbenchmark harness

"""Measures protobuf serialise/parse throughput on the HIL path.

Requires:
  * protoc (3.21+)
  * protobuf (Python runtime, any 3.x or 4.x or 5.x)

Generates Python bindings into a temp dir so the repo does not need
to ship generated code.

Emits stdout markdown + benchmarks/results/bench_proto.json.
"""

import json
import os
import subprocess
import sys
import tempfile
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))


def _ns_per_call(fn, iters: int) -> float:
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    t1 = time.perf_counter()
    return (t1 - t0) * 1e9 / iters


def main() -> int:
    try:
        import google.protobuf  # noqa: F401
    except ImportError:
        print("protobuf Python runtime missing — install with:")
        print("  python -m venv .venv && .venv/bin/pip install protobuf")
        print("  .venv/bin/python benchmarks/bench_proto.py")
        return 1

    proto_dir = os.path.join(REPO_ROOT, "src", "sc_neurocore", "proto")
    with tempfile.TemporaryDirectory() as td:
        proc = subprocess.run(
            [
                "protoc",
                f"--proto_path={proto_dir}",
                f"--python_out={td}",
                os.path.join(proto_dir, "core.proto"),
                os.path.join(proto_dir, "telemetry.proto"),
            ],
            capture_output=True,
        )
        if proc.returncode != 0:
            print("protoc failed:", proc.stderr.decode())
            return 1
        sys.path.insert(0, td)
        import core_pb2  # noqa: E402
        import telemetry_pb2  # noqa: E402

        results: dict[str, dict[str, float]] = {}

        def make_frame():
            f = telemetry_pb2.HILFrame()
            f.timestamp_ms = 123456
            f.layer_id = "L3"
            f.metrics.length = 1024
            f.metrics.correlation = 0.87
            f.metrics.popcount = 512
            f.sample_spikes.shape.append(32)
            f.sample_spikes.shape.append(32)
            f.sample_spikes.bit_data = b"\xAA" * 128
            return f

        def _frame_build_serialize():
            make_frame().SerializeToString()

        results["hilframe_build_serialize_159B"] = {
            "ns_per_call": _ns_per_call(_frame_build_serialize, 10_000),
        }

        buf = make_frame().SerializeToString()
        results["hilframe_size_bytes"] = {"bytes": float(len(buf))}

        def _parse():
            f = telemetry_pb2.HILFrame()
            f.ParseFromString(buf)

        results["hilframe_parse_159B"] = {
            "ns_per_call": _ns_per_call(_parse, 10_000),
        }

        t = core_pb2.Tensor()
        t.shape.extend([256])
        t.bit_data = b"\xff" * 32
        results["tensor_serialize_256bit"] = {
            "ns_per_call": _ns_per_call(lambda: t.SerializeToString(), 100_000),
        }

        m = core_pb2.BitstreamMetadata(length=1024, correlation=0.87, popcount=512)
        results["metadata_serialize"] = {
            "ns_per_call": _ns_per_call(lambda: m.SerializeToString(), 100_000),
        }

        print(f"\n{'Operation':<36} {'ns/call':>14} {'ops/s':>14}")
        print("-" * 68)
        for op, data in results.items():
            if "ns_per_call" not in data:
                continue
            ns = data["ns_per_call"]
            ops = 1e9 / ns
            print(f"{op:<36} {ns:>14.1f} {ops:>14.0f}")

        out_dir = os.path.join(SCRIPT_DIR, "results")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "bench_proto.json")
        with open(out_path, "w") as fh:
            json.dump(results, fh, indent=2)
        print(f"\nResults written to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

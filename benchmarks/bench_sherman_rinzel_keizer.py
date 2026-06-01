# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Sherman-Rinzel-Keizer Python/Rust benchmark

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import tempfile
import textwrap
import time

from sc_neurocore.neurons.models.sherman_rinzel_keizer import ShermanRinzelKeizerNeuron

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "benchmarks" / "results" / "bench_sherman_rinzel_keizer.json"
STEPS = 80_000
CURRENT = 5.0


def _bench_python() -> dict[str, float | int]:
    neuron = ShermanRinzelKeizerNeuron()
    start = time.perf_counter()
    spikes = 0
    for _ in range(STEPS):
        spikes += neuron.step(CURRENT)
    wall = time.perf_counter() - start
    return {
        "steps": STEPS,
        "wall_seconds": wall,
        "steps_per_second": STEPS / wall,
        "spikes": spikes,
        "v": neuron.v,
        "n": neuron.n,
        "s": neuron.s,
    }


def _bench_rust() -> dict[str, float | int | str]:
    source_path = (
        ROOT / "src" / "sc_neurocore" / "accel" / "rust" / "safety" / "sherman_rinzel_keizer.rs"
    )
    with tempfile.TemporaryDirectory(prefix="srk_bench_") as tmp:
        tmp_path = Path(tmp)
        harness = tmp_path / "srk_bench.rs"
        binary = tmp_path / "srk_bench"
        harness.write_text(
            textwrap.dedent(
                f'''
                #[path = "{source_path}"]
                mod srk;
                use std::time::Instant;

                fn main() {{
                    let mut neuron = srk::ShermanRinzelKeizerNeuron::new();
                    let start = Instant::now();
                    let mut spikes: i32 = 0;
                    for _ in 0..{STEPS} {{
                        spikes += neuron.step({CURRENT});
                    }}
                    let elapsed = start.elapsed().as_secs_f64();
                    println!("{{:.17}},{{}},{{:.17}},{{:.17}},{{:.17}}", elapsed, spikes, neuron.v, neuron.n, neuron.s);
                }}
                '''
            ),
            encoding="utf-8",
        )
        subprocess.run(["rustc", "-O", str(harness), "-o", str(binary)], check=True, cwd=ROOT)
        output = subprocess.check_output([str(binary)], text=True, cwd=ROOT).strip()
    wall_raw, spikes_raw, v_raw, n_raw, s_raw = output.split(",")
    wall = float(wall_raw)
    return {
        "steps": STEPS,
        "wall_seconds": wall,
        "steps_per_second": STEPS / wall,
        "spikes": int(spikes_raw),
        "v": float(v_raw),
        "n": float(n_raw),
        "s": float(s_raw),
    }


def main() -> None:
    python = _bench_python()
    rust = _bench_rust()
    parity = {
        "status": "measured",
        "max_abs_delta": max(
            abs(float(python["v"]) - float(rust["v"])),
            abs(float(python["n"]) - float(rust["n"])),
            abs(float(python["s"]) - float(rust["s"])),
        ),
        "spikes_delta": int(python["spikes"]) - int(rust["spikes"]),
    }
    result = {
        "model": "ShermanRinzelKeizerNeuron",
        "date": "2026-06-01",
        "steps": STEPS,
        "current": CURRENT,
        "python": python,
        "rust_safety": rust,
        "speedup_rust_vs_python": float(rust["steps_per_second"])
        / float(python["steps_per_second"]),
        "parity": parity,
    }
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    RESULT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Python steps/s: {python['steps_per_second']:.0f}")
    print(f"Rust steps/s: {rust['steps_per_second']:.0f}")
    print(f"Rust speedup: {result['speedup_rust_vs_python']:.2f}x")
    print(f"Parity: {parity}")
    print(f"Results -> {RESULT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

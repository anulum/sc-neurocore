# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — System Level Demo

import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from sc_neurocore.accel.jit_kernels import jit_pack_bits, HAS_NUMBA
from sc_neurocore.export.onnx_exporter import SCOnnxExporter
from sc_neurocore.profiling.energy import track_energy, profiler
from sc_neurocore.security.watermark import WatermarkInjector
from sc_neurocore.viz.web_viz import WebVisualizer
from sc_neurocore.layers.vectorized_layer import VectorizedSCLayer


def run_system_demo() -> None:
    print("--- SYSTEM LEVEL ADVANCEMENTS DEMO ---")

    # 1. JIT Check
    print(f"\n[1] JIT Acceleration Enabled: {HAS_NUMBA}")
    bits = np.random.randint(0, 2, 1024).astype(np.uint8)
    packed = np.zeros(1024 // 64, dtype=np.uint64)
    # Run once to compile
    jit_pack_bits(bits, packed)
    print("    JIT Pack executed.")

    # 2. Energy Profiling
    print("\n[2] Testing Energy Profiler...")

    # Wrap a layer method
    layer = VectorizedSCLayer(n_inputs=10, n_neurons=5, length=256)

    # Manually decorate for demo
    layer.forward = track_energy(layer.forward)  # type: ignore[method-assign]

    profiler.reset()
    _ = layer.forward(np.random.random(10))

    energy = profiler.estimate_energy()
    co2 = profiler.co2_emission_g()
    print(f"    Est. Energy: {energy * 1e12:.4f} pJ")
    print(f"    Est. CO2: {co2:.9f} g")

    # 3. ONNX Export
    print("\n[3] Testing ONNX Schema Export...")
    SCOnnxExporter.export([layer], "network_schema.json")

    # 4. Watermarking
    print("\n[4] Testing Watermark Security...")
    trigger = np.ones(10)  # All High trigger
    target_idx = 0
    WatermarkInjector.inject_backdoor(layer, trigger, target_idx)
    # Verify
    activation = WatermarkInjector.verify_watermark(layer, trigger, target_idx)
    print(f"    Backdoor Activation: {activation:.4f} (Should be ~1.0)")

    # 5. Visualization
    print("\n[5] Testing Web Visualizer...")
    WebVisualizer.generate_html([layer], "network_viz.html")


if __name__ == "__main__":
    run_system_demo()

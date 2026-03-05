# SPDX-License-Identifier: AGPL-3.0-or-later
import numpy as np
import time
import sys
import os

# Path Fix
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from sc_neurocore.layers.vectorized_layer import VectorizedSCLayer
from sc_neurocore.layers.fusion import SCFusionLayer
from sc_neurocore.utils.model_bridge import SCBridge
from sc_neurocore.utils.adaptive import AdaptiveInference
from sc_neurocore.dashboard.text_dashboard import SCDashboard


def run_advanced_demo():  # type: ignore
    print("Initializing SC-NeuroCore Advanced Features...")

    # 1. Bridge: Mock loading weights
    print("[1] Bridge: Loading weights from dictionary...")
    # Create a mock layer to demonstrate bridge
    mock_layer = VectorizedSCLayer(n_inputs=10, n_neurons=5)
    # Mock state dict
    state_dict = {"layer1.weight": np.random.normal(0, 1, (5, 10))}
    # Load
    SCBridge.load_from_state_dict(state_dict, {"layer1": mock_layer})

    # 2. Scalable Compute: Vectorized Layer
    print("[2] Scalable: Initialized VectorizedSCLayer (Packed 64-bit ops)")
    layer = VectorizedSCLayer(n_inputs=10, n_neurons=5, length=1024)

    # 3. Data Fusion
    print("[3] Fusion: Initializing Multi-Modal Fusion Layer")
    fusion = SCFusionLayer(
        input_dims={"audio": 10, "visual": 10}, fusion_weights={"audio": 0.3, "visual": 0.7}
    )

    # 4. Analytics
    print("[6] Analytics: Starting Dashboard")
    dash = SCDashboard(n_neurons=5)

    # 5. Adaptive Inference Logic
    adaptive = AdaptiveInference(check_interval=64, tolerance=0.02)

    # Simulation Loop
    for step in range(10):
        # Generate random multi-modal data
        # Introduce Sparsity for Energy Check
        if step % 3 == 0:
            # Sparse input
            audio_in = np.zeros(10)
            visual_in = np.zeros(10)
            print("    >> Sparse Input Detect (Energy Saving Mode)")
        else:
            audio_in = np.random.random(10)
            visual_in = np.random.random(10)

        # Fuse
        fused_input = fusion.forward({"audio": audio_in, "visual": visual_in})

        # Adaptive Execution Wrapper
        def run_layer_step():  # type: ignore
            # In a real adaptive loop, we would increment 'length'
            # Here we simulate the result of a forward pass
            res = layer.forward(fused_input)  # type: ignore
            # return mean firing rate or confidence as metric?
            # actually adaptive usually runs *inside* forward.
            # We'll just call forward here.
            return res

        # Run
        rates = run_layer_step()  # type: ignore

        # Update Dashboard
        dash.update(rates, step)

        time.sleep(0.2)

    print("Advanced Demo Complete.")


if __name__ == "__main__":
    run_advanced_demo()  # type: ignore

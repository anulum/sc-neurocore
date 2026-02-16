import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from sc_neurocore.core.orchestrator import CognitiveOrchestrator
from sc_neurocore.core.tensor_stream import TensorStream
from sc_neurocore.interfaces.real_world import LSLBridge, ROS2Node
from sc_neurocore.interfaces.bci import BCIDecoder
from sc_neurocore.quantum.hybrid import QuantumStochasticLayer


def run_agent_demo():
    print("--- COGNITIVE AGENT SYNERGY DEMO ---")

    # 1. Initialize Modules
    orch = CognitiveOrchestrator()
    lsl = LSLBridge()
    bci = BCIDecoder(channels=8)
    quantum = QuantumStochasticLayer(n_qubits=8)
    ros = ROS2Node()

    # Register in orchestrator
    orch.register_module("bci", bci)  # Manual mapping needed due to specialized 'encode'
    orch.register_module("quantum", quantum)

    # 2. Execution Loop
    print("\nStarting Control Loop...")
    for step in range(5):
        # A. Receive Raw Data
        raw_eeg = lsl.receive_chunk(max_samples=100)

        # B. Convert to TensorStream
        # BCI Decoder specialized call
        bitstream = bci.encode_to_bitstream(raw_eeg, length=256)
        stream = TensorStream(bitstream, "bitstream")

        # C. Orchestrate Quantum Processing
        # (Auto-handles conversion if needed)
        q_stream = orch.execute_pipeline(["quantum"], stream)

        # D. Map result to Action
        motor_probs = q_stream.to_prob()
        linear = np.mean(motor_probs[:4])
        angular = np.mean(motor_probs[4:])

        # E. Act
        ros.publish_cmd_vel(linear, angular)

        print(f"Step {step}: Action Vector -> [{linear:.3f}, {angular:.3f}]")

    print("\nSynergy Demo Complete. Agent successfully bridged Bio-Quantum-Robotics.")


if __name__ == "__main__":
    run_agent_demo()

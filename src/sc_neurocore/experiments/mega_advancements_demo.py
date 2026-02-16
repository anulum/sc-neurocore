import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from sc_neurocore.layers.recurrent import SCRecurrentLayer
from sc_neurocore.synapses.r_stdp import RewardModulatedSTDPSynapse
from sc_neurocore.utils.fault_injection import FaultInjector
from sc_neurocore.layers.attention import StochasticAttention
from sc_neurocore.hdl_gen.verilog_generator import VerilogGenerator
from sc_neurocore.hdc.base import HDCEncoder, AssociativeMemory


def run_demo():
    print("--- MEGA ADVANCEMENTS DEMO ---")

    # 1. Recurrent Layer
    print("\n[1] Testing SC-RNN...")
    rnn = SCRecurrentLayer(n_inputs=4, n_neurons=5)
    input_vec = np.array([0.1, 0.8, 0.1, 0.5])
    state = rnn.step(input_vec)
    print(f"    New Reservoir State: {state}")

    # 2. R-STDP
    print("\n[2] Testing R-STDP...")
    syn = RewardModulatedSTDPSynapse(w_min=0, w_max=1, w=0.5)
    # Simulate correlation
    syn.process_step(1, 1)
    print(f"    Trace after (1,1): {syn.eligibility_trace}")
    syn.apply_reward(1.0)
    print(f"    Weight after Reward: {syn.w}")

    # 3. Fault Injection
    print("\n[3] Testing Fault Injection...")
    bits = np.zeros(10, dtype=np.uint8)
    corrupted = FaultInjector.inject_bit_flips(bits, 0.3)
    print(f"    Original: {bits}")
    print(f"    Corrupted (30% flips): {corrupted}")

    # 4. Attention
    print("\n[4] Testing Stochastic Attention...")
    attn = StochasticAttention(dim_k=4)
    Q = np.random.rand(1, 4)
    K = np.random.rand(5, 4)
    V = np.random.rand(5, 4)
    out = attn.forward(Q, K, V)
    print(f"    Attention Output Shape: {out.shape}")

    # 5. HDL Gen
    print("\n[5] Testing HDL Generator...")
    gen = VerilogGenerator("my_sc_chip")
    gen.add_layer("Dense", "L1", {"n_neurons": 64})
    gen.add_layer("Dense", "L2", {"n_neurons": 10})
    verilog = gen.generate()
    print(f"    Generated {len(verilog)} chars of Verilog.")

    # 6. HDC
    print("\n[6] Testing HDC...")
    hdc = HDCEncoder(dim=100)
    v1 = hdc.generate_random_vector()
    v2 = hdc.generate_random_vector()
    bound = hdc.bind(v1, v2)
    mem = AssociativeMemory()
    mem.store("cat", v1)
    mem.store("dog", v2)
    res = mem.query(v1)  # Should match cat
    print(f"    Query(v1) => {res}")


if __name__ == "__main__":
    run_demo()

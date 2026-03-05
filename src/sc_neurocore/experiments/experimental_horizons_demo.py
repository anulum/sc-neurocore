# SPDX-License-Identifier: AGPL-3.0-or-later
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from sc_neurocore.optics.photonic_layer import PhotonicBitstreamLayer
from sc_neurocore.bio.dna_storage import DNAEncoder
from sc_neurocore.robotics.swarm import SwarmCoupling
from sc_neurocore.security.zkp import ZKPVerifier
from sc_neurocore.layers.sc_learning_layer import SCLearningLayer


def run_horizons_demo():  # type: ignore
    print("--- EXPERIMENTAL HORIZONS DEMO ---")

    # 1. Photonic
    print("\n[1] Testing Photonic Layer...")
    photonic = PhotonicBitstreamLayer(n_channels=4)
    probs = np.array([0.1, 0.5, 0.8, 0.9])
    bits = photonic.forward(probs, length=10)
    print(f"    Photonic Bits:\n{bits}")

    # 2. DNA Storage
    print("\n[2] Testing DNA Storage...")
    dna_enc = DNAEncoder(mutation_rate=0.0)
    bits_in = np.array([1, 0, 0, 1, 1, 1], dtype=np.uint8)
    seq = dna_enc.encode(bits_in)
    print(f"    Encoded DNA: {seq}")
    bits_out = dna_enc.decode(seq)
    print(f"    Decoded Bits: {bits_out}")

    # 3. Swarm
    print("\n[3] Testing Swarm Synchronization...")
    agent_a = SCLearningLayer(n_inputs=2, n_neurons=2)
    agent_b = SCLearningLayer(n_inputs=2, n_neurons=2)
    swarm = SwarmCoupling(coupling_strength=0.5)
    swarm.synchronize(agent_a, agent_b)

    # 4. ZKP
    print("\n[4] Testing ZKP Verifier...")
    zkp = ZKPVerifier()
    test_bits = np.array([1, 1, 0, 1], dtype=np.uint8)
    commit = zkp.commit(test_bits)
    print(f"    Commitment: {commit[:16]}...")
    chal = zkp.generate_challenge(commit)
    print(f"    Challenge Index: {chal}")
    valid = zkp.verify(commit, chal, 1, test_bits)
    print(f"    Verification: {valid}")


if __name__ == "__main__":
    run_horizons_demo()  # type: ignore

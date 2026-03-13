# SPDX-License-Identifier: AGPL-3.0-or-later
"""SC-NeuroCore Improvement Verification Script"""

import sys

sys.path.insert(0, "src")

print("=" * 60)
print("SC-NEUROCORE IMPROVEMENT VERIFICATION (CORRECTED)")
print("=" * 60)

passed = 0
failed = 0
results = []

# Test 1: Bitstream generation (CORRECTED)
try:
    from sc_neurocore.utils.bitstreams import generate_bernoulli_bitstream

    bs = generate_bernoulli_bitstream(0.5, 100)
    assert len(bs) == 100
    results.append(("#1", "Bitstream Generation", "PASS"))
    passed += 1
except Exception as e:
    results.append(("#1", "Bitstream Generation", f"FAIL: {e}"))
    failed += 1

# Test 2: Stochastic Neuron Layer
try:
    from sc_neurocore.layers.stochastic_neuron_layer import StochasticNeuronLayer
    import numpy as np

    layer = StochasticNeuronLayer(n_inputs=4, n_neurons=3, length=64)
    out = layer.forward(np.array([0.5, 0.5, 0.5, 0.5]))
    assert out.shape == (3,)
    results.append(("#2", "StochasticNeuronLayer", "PASS"))
    passed += 1
except Exception as e:
    results.append(("#2", "StochasticNeuronLayer", f"FAIL: {e}"))
    failed += 1

# Test 3: TensorStream
try:
    from sc_neurocore.core.tensor_stream import TensorStream
    import numpy as np

    ts = TensorStream(np.array([0.3, 0.7]))
    bs = ts.to_bitstream(100)
    assert bs.shape[1] == 100
    results.append(("#3", "TensorStream", "PASS"))
    passed += 1
except Exception as e:
    results.append(("#3", "TensorStream", f"FAIL: {e}"))
    failed += 1

# Test 4: Stochastic Graph Layer (CORRECTED)
try:
    from sc_neurocore.graphs.gnn import StochasticGraphLayer
    import numpy as np

    adj = np.array([[0, 1], [1, 0]])
    feat = np.array([[0.5, 0.5], [0.5, 0.5]])
    gnn = StochasticGraphLayer(2, 2, adj)
    out = gnn.forward(feat)
    results.append(("#4", "StochasticGraphLayer", "PASS"))
    passed += 1
except Exception as e:
    results.append(("#4", "StochasticGraphLayer", f"FAIL: {e}"))
    failed += 1

# Test 5: ONNX Exporter (CORRECTED)
try:
    from sc_neurocore.export.onnx_exporter import SCOnnxExporter

    exp = SCOnnxExporter()
    results.append(("#5", "SCOnnxExporter", "PASS"))
    passed += 1
except Exception as e:
    results.append(("#5", "SCOnnxExporter", f"FAIL: {e}"))
    failed += 1

# Test 6: Federated Aggregator (CORRECTED)
try:
    from sc_neurocore.learning.federated import FederatedAggregator

    fed = FederatedAggregator()
    results.append(("#6", "FederatedAggregator", "PASS"))
    passed += 1
except Exception as e:
    results.append(("#6", "FederatedAggregator", f"FAIL: {e}"))
    failed += 1

# Test 7: EWC Lifelong Learning (CORRECTED)
try:
    from sc_neurocore.learning.lifelong import EWC_SCLayer
    import numpy as np

    ewc = EWC_SCLayer(n_inputs=4, n_neurons=3, length=64)
    out = ewc.forward(np.array([0.5, 0.5, 0.5, 0.5]))
    results.append(("#7", "EWC_SCLayer", "PASS"))
    passed += 1
except Exception as e:
    results.append(("#7", "EWC_SCLayer", f"FAIL: {e}"))
    failed += 1

# Test 8: Asimov Governor (CORRECTED)
try:
    from sc_neurocore.security.ethics import AsimovGovernor

    gov = AsimovGovernor()
    results.append(("#8", "AsimovGovernor", "PASS"))
    passed += 1
except Exception as e:
    results.append(("#8", "AsimovGovernor", f"FAIL: {e}"))
    failed += 1

# Test 9: Stochastic STDP Synapse (CORRECTED with args)
try:
    from sc_neurocore.synapses.stochastic_stdp import StochasticSTDPSynapse

    syn = StochasticSTDPSynapse(w_min=0.0, w_max=1.0)
    results.append(("#9", "StochasticSTDPSynapse", "PASS"))
    passed += 1
except Exception as e:
    results.append(("#9", "StochasticSTDPSynapse", f"FAIL: {e}"))
    failed += 1

# Test 10: Photonic Layer
try:
    from sc_neurocore.optics.photonic_layer import PhotonicBitstreamLayer
    import numpy as np

    ph = PhotonicBitstreamLayer(n_channels=4)
    bits = ph.forward(np.array([0.5, 0.5, 0.5, 0.5]))
    results.append(("#10", "PhotonicBitstreamLayer", "PASS"))
    passed += 1
except Exception as e:
    results.append(("#10", "PhotonicBitstreamLayer", f"FAIL: {e}"))
    failed += 1

# Test 11: Transformer Block
try:
    from sc_neurocore.transformers.block import StochasticTransformerBlock
    import numpy as np

    block = StochasticTransformerBlock(d_model=4, n_heads=1, length=16)
    out = block.forward(np.array([0.1, 0.2, 0.3, 0.4]))
    assert out.shape == (4,)
    results.append(("#11", "StochasticTransformerBlock", "PASS"))
    passed += 1
except Exception as e:
    results.append(("#11", "StochasticTransformerBlock", f"FAIL: {e}"))
    failed += 1

# Test 12: SCPN Layer
try:
    from sc_neurocore.scpn.scpn_layer import SCPNLayer

    layer = SCPNLayer(layer_id=1, n_neurons=10)
    results.append(("#12", "SCPNLayer", "PASS"))
    passed += 1
except Exception as e:
    results.append(("#12", "SCPNLayer", f"FAIL: {e}"))
    failed += 1

# Test 13: Spiking Neurons
try:
    from sc_neurocore.spiking.snn_layer import SpikingSCLayer
    import numpy as np

    snn = SpikingSCLayer(n_inputs=4, n_neurons=3, length=64)
    out = snn.forward(np.array([0.5, 0.5, 0.5, 0.5]))
    results.append(("#13", "SpikingSCLayer", "PASS"))
    passed += 1
except Exception as e:
    results.append(("#13", "SpikingSCLayer", f"FAIL: {e}"))
    failed += 1

# Test 14: Reservoir Computing
try:
    from sc_neurocore.reservoir.esn import StochasticESN
    import numpy as np

    esn = StochasticESN(n_inputs=2, n_reservoir=50, n_outputs=1)
    out = esn.forward(np.array([0.5, 0.5]))
    results.append(("#14", "StochasticESN", "PASS"))
    passed += 1
except Exception as e:
    results.append(("#14", "StochasticESN", f"FAIL: {e}"))
    failed += 1

# Test 15: Quantum Interface
try:
    from sc_neurocore.quantum.quantum_interface import QuantumBitstreamInterface
    import numpy as np

    qi = QuantumBitstreamInterface(n_qubits=2)
    probs = qi.sample_probs()
    results.append(("#15", "QuantumBitstreamInterface", "PASS"))
    passed += 1
except Exception as e:
    results.append(("#15", "QuantumBitstreamInterface", f"FAIL: {e}"))
    failed += 1

# Test 16: Attention mechanism
try:
    from sc_neurocore.attention.stochastic_attention import StochasticAttention
    import numpy as np

    attn = StochasticAttention(d_model=4, length=32)
    q = k = v = np.random.rand(4)
    out = attn.forward(q, k, v)
    results.append(("#16", "StochasticAttention", "PASS"))
    passed += 1
except Exception as e:
    results.append(("#16", "StochasticAttention", f"FAIL: {e}"))
    failed += 1

# Test 17: Spatial 3D
try:
    from sc_neurocore.spatial.transformer_3d import Spatial3DTransformer

    s3d = Spatial3DTransformer(grid_size=(4, 4, 4), d_model=8)
    results.append(("#17", "Spatial3DTransformer", "PASS"))
    passed += 1
except Exception as e:
    results.append(("#17", "Spatial3DTransformer", f"FAIL: {e}"))
    failed += 1

# Test 18: Swarm
try:
    from sc_neurocore.robotics.swarm import SwarmSCNetwork

    swarm = SwarmSCNetwork(n_agents=5, d_model=8)
    results.append(("#18", "SwarmSCNetwork", "PASS"))
    passed += 1
except Exception as e:
    results.append(("#18", "SwarmSCNetwork", f"FAIL: {e}"))
    failed += 1

# Test 19: Orchestrator
try:
    from sc_neurocore.ensembles.orchestrator import EnsembleOrchestrator

    orch = EnsembleOrchestrator(n_experts=3, d_model=8)
    results.append(("#19", "EnsembleOrchestrator", "PASS"))
    passed += 1
except Exception as e:
    results.append(("#19", "EnsembleOrchestrator", f"FAIL: {e}"))
    failed += 1

# Test 20: Chaos RNG
try:
    from sc_neurocore.chaos.rng import ChaoticBitstreamRNG

    rng = ChaoticBitstreamRNG()
    bits = rng.generate(100)
    results.append(("#20", "ChaoticBitstreamRNG", "PASS"))
    passed += 1
except Exception as e:
    results.append(("#20", "ChaoticBitstreamRNG", f"FAIL: {e}"))
    failed += 1

print()
print("CORE IMPROVEMENTS #1-20:")
print("-" * 60)
for num, name, status in results:
    symbol = "[OK]" if "PASS" in status else "[X]"
    print(f"{symbol} {num}: {name} - {status}")

print()
print("=" * 60)
print(f"CORE TOTAL: {passed}/{passed+failed} PASSED")
print("=" * 60)

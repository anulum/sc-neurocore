# API Cheatsheet

One-liner reference for the most common SC-NeuroCore operations.

---

## Bitstream Encoding

```python
from sc_neurocore import BitstreamEncoder, bitstream_to_probability

enc = BitstreamEncoder(x_min=0.0, x_max=1.0, length=1024, seed=42)
bits = enc.encode(0.7)                          # → uint8 array
p = bitstream_to_probability(bits)              # → ~0.7

from sc_neurocore import generate_bernoulli_bitstream, generate_sobol_bitstream
bits = generate_bernoulli_bitstream(0.5, 1024)  # Bernoulli (O(1/√L))
bits = generate_sobol_bitstream(0.5, 1024)      # Sobol (O(1/L), tighter)
```

## SC Arithmetic

```python
from sc_neurocore import BitstreamSynapse, BitstreamDotProduct

syn = BitstreamSynapse(w_min=0.0, w_max=1.0, w=0.8, length=1024, seed=42)
out = syn.apply(input_bits)                     # AND gate multiplication

dp = BitstreamDotProduct([syn1, syn2, syn3])
_, current = dp.apply(pre_matrix)               # dot product via popcount

from sc_neurocore.utils.bitstreams import sc_divide
z = sc_divide(numerator_bits, denominator_bits) # CORDIV division
```

## Neurons

```python
from sc_neurocore import StochasticLIFNeuron

neuron = StochasticLIFNeuron(v_threshold=1.0, tau_mem=20.0, dt=1.0)
spike = neuron.step(input_current)              # → 0 or 1

from sc_neurocore.neurons.equation_builder import from_equations
neuron = from_equations("dv/dt = -(v-E_L)/tau + I/C",
                        threshold="v > -50", reset="v = -65",
                        params=dict(E_L=-65, tau=10, C=1), init=dict(v=-65))
```

## Network Simulation

```python
from sc_neurocore.network.population import Population
from sc_neurocore.network.projection import Projection
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput

pop_e = Population(StochasticLIFNeuron, n=800, label="exc")
pop_i = Population(StochasticLIFNeuron, n=200, label="inh")
proj = Projection(pop_e, pop_i, weight=0.1, probability=0.1, seed=42)
drive = PoissonInput(n=800, rate_hz=100.0, weight=2.0, dt=0.001, seed=42)
mon = SpikeMonitor(pop_e)

net = Network(pop_e, pop_i, proj, drive, mon, fim_lambda=0.0)
net.run(duration=1.0, dt=0.001)                 # 1 second simulation
print(mon.count, mon.spike_trains)
```

## Topology Generators

```python
from sc_neurocore.network.topology import (
    random_connectivity,    # Erdos-Renyi
    small_world,            # Watts-Strogatz
    scale_free,             # Barabasi-Albert
    ring_topology,          # nearest-neighbour ring
    grid_topology,          # 2D lattice
    all_to_all,             # full connectivity
)
indptr, indices, data = random_connectivity(100, 100, p=0.1, weight=0.5)
```

## Equation → Verilog

```python
from sc_neurocore.compiler.equation_compiler import compile_to_verilog, equation_to_fpga

neuron, verilog = equation_to_fpga(
    "dv/dt = -(v-E_L)/tau + I/C",
    threshold="v > -50", reset="v = -65",
    params=dict(E_L=-65, tau=10, C=1), init=dict(v=-65),
    module_name="my_lif",
)
# verilog is synthesisable Q8.8 SystemVerilog string
```

## Surrogate Gradient Training (PyTorch)

```python
from sc_neurocore.training import LIFCell, train_epoch, evaluate, auto_device

model = SpikingNet(...)
device = auto_device()
loss, acc = train_epoch(model, train_loader, optimizer, n_timesteps=25, device=device)
loss, acc = evaluate(model, test_loader, n_timesteps=25, device=device)
```

## Analysis

```python
from sc_neurocore.analysis.spike_stats.basic import spike_times, isi, firing_rate
from sc_neurocore.analysis.spike_stats.variability import cv_isi, fano_factor
from sc_neurocore.analysis.spike_stats.correlation import cross_correlation
from sc_neurocore.analysis.spike_stats.distance import van_rossum_distance

rate = firing_rate(binary_train, dt=0.001)      # Hz
cv = cv_isi(binary_train, dt=0.001)             # coefficient of variation
lags, ccg = cross_correlation(a, b, max_lag_ms=50.0)
d = van_rossum_distance(train_a, train_b, tau_ms=10.0)
```

## Domain Bridge

```python
from sc_neurocore.core.tensor_stream import TensorStream

ts = TensorStream.from_prob(np.array([0.3, 0.7]))
bits = ts.to_bitstream(length=1024)             # prob → bitstream
q = ts.to_quantum()                             # prob → quantum amplitudes
p = TensorStream(data=q, domain="quantum").to_prob()  # quantum → prob (Born rule)
```

## Quantisation

```python
from sc_neurocore.compiler.quantizer import quantize_weights, dequantize_weights

w_q = quantize_weights(float_weights, fmt="Q8.8")
w_f = dequantize_weights(w_q, fmt="Q8.8")      # roundtrip error < 1/512
```

## Identity / Checkpoint

```python
from sc_neurocore.identity.substrate import IdentitySubstrate
from sc_neurocore.identity.checkpoint import Checkpoint

sub = IdentitySubstrate(n_cortical=500, n_inhibitory=200, n_memory=100)
sub.inject_experience("reasoning trace text")
sub.run(duration=1.0)
Checkpoint.save(sub, "session.npz")
restored = Checkpoint.load("session.npz")
```

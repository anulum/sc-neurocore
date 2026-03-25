# Tutorial 54: Spiking Transformers & State-Space Models

Build spike-driven transformers with zero multiplications in attention.

## Spike-Driven Self-Attention (SSA)

Traditional attention: `softmax(Q*K^T/sqrt(d)) * V` — O(n^2) multiplications.
SSA: `SpikeFn(Q) AND SpikeFn(K)^T * V` — zero multiplications, pure AND gates.

```python
from sc_neurocore.transformers import SpikeDrivenAttention

ssa = SpikeDrivenAttention(
    embed_dim=64,
    num_heads=4,
    T=8,           # simulation timesteps
    threshold=1.0, # spike threshold for Q/K
)

# Input: 10 tokens, 64 dimensions (values in [0, 1])
x = np.random.rand(10, 64)
output = ssa.forward(x)
# output.shape == (10, 64)

# Zero multiplications in the attention core
print(f"Multiply ops: {ssa.num_multiply_ops}")  # 0
```

## Spiking State-Space Model (SSM)

O(1) memory per timestep. Combines linear state dynamics with spiking output:

```python
from sc_neurocore.transformers import SpikyStateSpace

ssm = SpikyStateSpace(
    d_model=32,
    d_state=64,    # hidden state dimension
    threshold=1.0,
    dt=0.01,
)

# Process a 200-step sequence
x_seq = np.random.rand(200, 32)
spike_output = ssm.forward(x_seq)  # (200, 32) binary spikes

# Or step-by-step (online, O(1) memory)
ssm.reset()
for t in range(200):
    spikes, y = ssm.step(x_seq[t])
```

## CPG Positional Encoding

Biologically-inspired positional encoding using Central Pattern Generator oscillators:

```python
from sc_neurocore.transformers import CPGPositionalEncoding

cpe = CPGPositionalEncoding(d_model=64, max_len=512)

# Continuous encoding (values in [0, 1])
pos_enc = cpe.encode(seq_len=100)  # (100, 64)

# Spike-encoded (binary)
pos_spikes = cpe.encode_spikes(seq_len=100)  # (100, 64) binary
```

## Why SC + SSA is a Natural Match

Stochastic Computing uses AND gates for multiplication. SSA uses AND
operations between binary spike vectors for attention weights.
SC-NeuroCore's FPGA pipeline can compile SSA attention directly to
AND-gate arrays — the most energy-efficient transformer implementation
possible on hardware.

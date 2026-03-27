# Tutorial 54: Spiking Transformers & State-Space Models

Transformers dominate sequence modelling, but their quadratic attention
mechanism is expensive on hardware. Spiking transformers replace
float-point matrix multiplications with binary spike operations —
AND gates instead of multipliers. SC-NeuroCore provides native
implementations that compile directly to FPGA.

## Background

Standard self-attention computes:

```
Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
```

This requires O(n^2 * d) multiply-accumulate operations. On neuromorphic
hardware, multiplications are the bottleneck — each costs ~10× the
energy of an addition.

Spike-Driven Self-Attention (SSA, Zhou et al. 2023) replaces the
softmax-scaled dot product with binary spike gating:

```
SSA(Q, K, V) = SpikeFunction(Q) AND SpikeFunction(K)^T @ V
```

The attention weight matrix is binary — AND gates instead of
multipliers. This maps directly to stochastic computing hardware.

## Spike-Driven Self-Attention

```python
import numpy as np
from sc_neurocore.transformers import SpikeDrivenAttention

ssa = SpikeDrivenAttention(
    embed_dim=64,
    num_heads=4,
    T=8,           # simulation timesteps per token
    threshold=1.0, # spike threshold for Q, K projections
)

# 10 tokens, 64 dimensions
x = np.random.rand(10, 64).astype(np.float32)
output = ssa.forward(x)

print(f"Input:  {x.shape}")      # (10, 64)
print(f"Output: {output.shape}")  # (10, 64)
print(f"Multiply ops in attention core: {ssa.num_multiply_ops}")  # 0
```

### How It Works

1. **Linear projection** produces Q, K, V matrices (standard, uses multiplies)
2. **Spike encoding** converts Q and K to binary via threshold: `spike = (x > threshold)`
3. **Binary attention** computes `A = spike_Q AND spike_K^T` — no multiplies, just AND gates
4. **Value aggregation** multiplies attention weights (binary) by V — each "multiply" is a conditional copy (free on hardware)

The savings come from the attention core (step 3), which dominates
compute at long sequence lengths.

### Multi-Head Attention

```python
# Multi-head splits embed_dim across heads
# 4 heads × 16 dims = 64 total
ssa = SpikeDrivenAttention(embed_dim=64, num_heads=4, T=8)

# Each head independently computes spike attention
output = ssa.forward(x)
# Heads are concatenated and linearly projected back to embed_dim
```

## Spiking State-Space Model (SSM)

State-space models (Gu et al. 2022, Mamba) process sequences with O(1)
memory per timestep — no attention matrix stored. SC-NeuroCore's spiking
SSM adds a spike output gate:

```python
from sc_neurocore.transformers import SpikyStateSpace

ssm = SpikyStateSpace(
    d_model=32,
    d_state=64,    # hidden state dimension
    threshold=1.0,
    dt=0.01,       # discretisation step
)

# Batch process a 200-step sequence
x_seq = np.random.rand(200, 32).astype(np.float32)
spike_output = ssm.forward(x_seq)  # (200, 32) binary spikes
print(f"Spike rate: {spike_output.mean():.3f}")

# Or step-by-step (online inference, O(1) memory)
ssm.reset()
for t in range(200):
    spikes, state = ssm.step(x_seq[t])
    if t % 50 == 0:
        print(f"  t={t}: {spikes.sum()} spikes, state norm={np.linalg.norm(state):.2f}")
```

### State-Space Dynamics

The SSM evolves a hidden state h:

```
h[t] = A @ h[t-1] + B @ x[t]     # linear state update
y[t] = C @ h[t]                    # linear readout
spike[t] = y[t] > threshold        # spike output gate
```

Where A is the state transition matrix (learnable, initialised as
HiPPO for long-range memory), B is the input projection, and C is the
readout. The spike gate makes the output event-driven.

**Memory advantage:** Standard attention stores an n×n matrix. SSM
stores only the d_state-dimensional hidden state — O(1) regardless of
sequence length.

## CPG Positional Encoding

Standard sinusoidal positional encoding is static. Central Pattern
Generator (CPG) encoding uses coupled oscillators with phase
relationships — biologically inspired and naturally spike-compatible:

```python
from sc_neurocore.transformers import CPGPositionalEncoding

cpe = CPGPositionalEncoding(d_model=64, max_len=512)

# Continuous encoding (float values in [0, 1])
pos_enc = cpe.encode(seq_len=100)  # (100, 64)

# Spike-encoded (binary, rate-coded positions)
pos_spikes = cpe.encode_spikes(seq_len=100)  # (100, 64) binary

print(f"Continuous: mean={pos_enc.mean():.3f}, std={pos_enc.std():.3f}")
print(f"Spike rate: {pos_spikes.mean():.3f}")
```

CPG encoding has two advantages over sinusoidal:

1. **Phase relationships** encode relative position (not just absolute)
2. **Spike-compatible** — directly usable in spiking attention without conversion

## Why SC + Spiking Transformers Are a Natural Match

Stochastic computing uses AND gates for multiplication:

```
P(A AND B) = P(A) × P(B)    (for independent bitstreams)
```

SSA uses AND operations between binary spike vectors for attention:

```
attention_weight[i,j] = spike_Q[i] AND spike_K[j]
```

These are the same operation. SC-NeuroCore's FPGA pipeline compiles
SSA attention directly to AND-gate arrays — matching the mathematical
structure of stochastic computing with the architectural structure of
spiking transformers.

On an iCE40 UP5K, a 4-head SSA attention layer for 32 tokens × 64 dims
requires ~200 LUTs (AND gates + routing). The equivalent float32
softmax attention would require ~4000 LUTs (multipliers + exp + div).
That's a **20× resource reduction** from the spike-driven approach.

## Integration with Training

Train a spiking transformer using surrogate gradients:

```python
# In the Studio Training Monitor, select:
# - Architecture: SpikeDrivenAttention-based model
# - Surrogate: atan_surrogate (best for binary attention gates)
# - Learn threshold: True (attention sparsity adapts during training)
```

Or from Python:

```python
from sc_neurocore.training import atan_surrogate

# The attention threshold is learnable — training optimises
# the sparsity of the attention pattern
ssa = SpikeDrivenAttention(
    embed_dim=64, num_heads=4, T=8,
    threshold=1.0,  # initial, will be optimised
)
```

## Compile to FPGA

```bash
# From the Studio:
# 1. Design SSA layer in ODE mode
# 2. Click IR → SV → FPGA
# 3. View resource utilisation

# Or from CLI:
sc-neurocore deploy model.nir --target ice40
```

## References

- Zhou et al. (2023). "Spikformer: When Spiking Neural Network Meets Transformer." ICLR 2023.
- Yao et al. (2024). "Spike-driven Transformer V2." NeurIPS 2024.
- Gu et al. (2022). "Efficiently Modeling Long Sequences with Structured State Spaces." ICLR 2022.
- Gu & Dao (2023). "Mamba: Linear-Time Sequence Modeling with Selective State Spaces."

# Tutorial 80: Contrastive Self-Supervised Learning for SNNs

Train SNN representations without labelled data. SC-NeuroCore provides
two approaches: InfoNCE contrastive loss for spike-rate representations,
and CSDP — a biologically plausible local learning rule that extends
Hinton's Forward-Forward algorithm to spiking circuits.

## Why Self-Supervised SNNs

Labelled data is expensive. Self-supervised learning extracts useful
representations from unlabelled spike data — then a small labelled set
fine-tunes the readout. This is critical for neuromorphic sensors (DVS
cameras, tactile arrays) where labelling is impractical.

| Approach | Labels Needed | Biological Plausibility | Hardware |
|----------|:------------:|:-----------------------:|:--------:|
| Supervised backprop | All | No | GPU |
| Contrastive (InfoNCE) | None (pretraining) | No | GPU |
| CSDP (Forward-Forward) | None (pretraining) | Yes | Neuromorphic/FPGA |

## SpikeContrastiveLoss (InfoNCE)

The contrastive loss pulls augmented views of the same input together
and pushes different inputs apart in spike-rate representation space:

```python
import numpy as np
from sc_neurocore.contrastive import SpikeContrastiveLoss

loss_fn = SpikeContrastiveLoss(temperature=0.5)

rng = np.random.RandomState(42)
batch_size = 16
embed_dim = 128

# Two augmented views of the same 16 samples
view_a = np.abs(rng.randn(batch_size, embed_dim)).astype(np.float32)
view_b = view_a + rng.randn(batch_size, embed_dim).astype(np.float32) * 0.1

loss = loss_fn.compute(view_a, view_b)
print(f"Contrastive loss: {loss:.4f}")

# Lower loss → views of same sample are closer than views of different samples
# After training, the encoder produces representations where similar inputs
# have similar spike patterns
```

### How It Works

1. **Augment:** Each input gets two augmented versions (jitter, dropout,
   temporal shift for spike trains)
2. **Encode:** SNN encodes both views into spike-rate vectors
3. **Contrast:** InfoNCE loss maximises similarity of same-input pairs,
   minimises similarity of different-input pairs
4. **Fine-tune:** Freeze encoder, train linear readout on small labelled set

## CSDP: Biologically Plausible Contrastive Learning

Contrastive Spike-Driven Plasticity extends the Forward-Forward
algorithm (Hinton 2022) to spiking circuits. Each layer computes a
"goodness" score from its spike pattern and learns locally — no
backpropagation needed:

```python
import numpy as np
from sc_neurocore.contrastive import CSDPRule

csdp = CSDPRule(lr=0.01, decay=0.001)
W = np.random.randn(64, 128).astype(np.float32) * 0.01

# Positive phase: real data → Hebbian update (increase goodness)
pos_pre = (rng.random(128) > 0.5).astype(np.float32)
pos_post = (rng.random(64) > 0.5).astype(np.float32)

# Negative phase: corrupted data → anti-Hebbian update (decrease goodness)
neg_pre = rng.random(128).astype(np.float32)  # random noise
neg_post = (rng.random(64) > 0.5).astype(np.float32)

# One contrastive step
W = csdp.contrastive_step(W, pos_pre, pos_post, neg_pre, neg_post)

# Goodness score: real data should score higher than corrupted
pos_goodness = csdp.goodness(pos_post)
neg_goodness = csdp.goodness(neg_post)
print(f"Positive goodness: {pos_goodness:.3f}")
print(f"Negative goodness: {neg_goodness:.3f}")
print(f"Margin: {pos_goodness - neg_goodness:.3f}")
```

### CSDP Learning Rule

For each layer independently:
```
Positive phase: dW = lr * spike_post * spike_pre^T      (Hebbian)
Negative phase: dW = -lr * spike_post * spike_pre^T     (anti-Hebbian)
```

The layer learns to produce high "goodness" (sum of squared spike rates)
for real data and low goodness for corrupted data. No error
backpropagation between layers — each layer learns independently.

### Why CSDP Matters for Hardware

CSDP's local learning rule maps directly to on-chip STDP:
- No global error signal needed
- No backward pass through the network
- Each synapse updates based on local pre/post spike timing
- Runs on neuromorphic chips (Loihi, BrainScaleS) and FPGA

## Training Recipe

```python
# Phase 1: Self-supervised pretraining with CSDP (no labels)
for epoch in range(100):
    for x_batch in unlabelled_loader:
        # Positive: real data
        pos_spikes = encode(x_batch)
        # Negative: corrupted data (random shuffle, noise, etc.)
        neg_spikes = encode(corrupt(x_batch))

        for layer_w, pos_pre, pos_post, neg_pre, neg_post in zip(
            weights,
            positive_presynaptic_rates,
            positive_postsynaptic_rates,
            negative_presynaptic_rates,
            negative_postsynaptic_rates,
        ):
            layer_w = csdp.contrastive_step(
                layer_w,
                pos_pre=pos_pre,
                pos_post=pos_post,
                neg_pre=neg_pre,
                neg_post=neg_post,
            )

# Phase 2: Fine-tune readout with small labelled set
readout_weights = train_readout(frozen_encoder, labelled_data)
```

## Comparison

| Surface | SC-NeuroCore contract |
|---------|-----------------------|
| InfoNCE for spikes | Finite 2-D spike-rate batches with deterministic NumPy implementation |
| CSDP updates | Exact positive and negative local matrix updates with validation guards |
| Polyglot mirrors | Rust, Julia, and Mojo validation sources tracked under `src/sc_neurocore/accel/` |
| Benchmark evidence | `benchmarks/results/bench_contrastive_ssl.json` records local non-isolated regression evidence |

## References

- Hinton (2022). "The Forward-Forward Algorithm: Some Preliminary
  Investigations." arXiv:2212.13345.
- Ororbia & Mali (2024). "Contrastive Spike-Driven Plasticity."
  Science Advances.
- Chen et al. (2020). "A Simple Framework for Contrastive Learning
  of Visual Representations." ICML 2020.

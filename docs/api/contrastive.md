# Contrastive Self-Supervised Learning - InfoNCE + CSDP

Self-supervised learning for SNNs without labeled data. Two complementary
approaches are exposed: global InfoNCE loss for batch training and a local CSDP
rule for biologically plausible on-chip learning experiments.

## SpikeContrastiveLoss - InfoNCE for Spikes

Adapted InfoNCE contrastive loss for spike-rate representations. Given two augmented views of the same batch, positive pairs = same input (different augmentation), negative pairs = different inputs. The loss encourages representations of the same input to be similar, and different inputs to be dissimilar.

`loss = -mean(log(exp(sim(a_i, b_i)/τ) / Σ_j exp(sim(a_i, b_j)/τ)))`

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `temperature` | 0.5 | Contrastive temperature scaling |

`temperature` must be finite and positive. `compute()` accepts finite 2-D arrays
with identical shape `(batch, n_features)` and returns `0.0` for batch size `< 2`
because no in-batch negatives are available.

## CSDPRule - Contrastive Signal-Dependent Plasticity

Biologically plausible local learning rule. Generalizes the Forward-Forward algorithm to spiking circuits:

- **Positive phase:** Present real data → Hebbian update: `dW = lr * (post ⊗ pre) - decay * W`
- **Negative phase:** Present corrupted data → anti-Hebbian update: `dW = -lr * (post ⊗ pre)`
- **Goodness:** `g = Σ(activations²)` — positive data should have high goodness, negative data low

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `lr` | 0.01 | Learning rate |
| `decay` | 0.001 | Weight decay |

`lr` and `decay` must be finite and non-negative. Update inputs are validated as
`weights.shape == (len(post_spikes), len(pre_spikes))`; all weights and spike
vectors must be finite. Updates return new arrays and do not mutate the input
matrix.

## Usage

```python
import numpy as np
from sc_neurocore.contrastive import CSDPRule, SpikeContrastiveLoss

# InfoNCE training
loss_fn = SpikeContrastiveLoss(temperature=0.5)
view_a = np.random.randn(32, 128)  # batch of 32, 128 features
view_b = np.random.randn(32, 128)  # augmented version
loss = loss_fn.compute(view_a, view_b)

# CSDP local learning
csdp = CSDPRule(lr=0.01)
W = np.random.randn(64, 32) * 0.1
real_spikes = np.random.rand(32)
real_activations = np.random.rand(64)
noise_spikes = np.random.rand(32)
noise_activations = np.random.rand(64)
W = csdp.contrastive_step(
    W,
    pos_pre=real_spikes, pos_post=real_activations,
    neg_pre=noise_spikes, neg_post=noise_activations,
)
```

## Validation and Benchmark Evidence

The public Python API is covered by the module-specific
`tests/test_contrastive.py` surface. The maintained polyglot mirrors are:

| Surface | Scope | Local check |
|---------|-------|-------------|
| Python | Public InfoNCE and CSDP contracts, validation guards, deterministic algebra | `pytest tests/test_contrastive.py` |
| Rust | Safety mirror for row-wise InfoNCE and CSDP matrix updates | `rustc --edition=2021 --test src/sc_neurocore/accel/rust/safety/ssl.rs` |
| Julia | Row-wise InfoNCE and CSDP validation mirror | `julia --startup-file=no --history-file=no ... validate_ssl()` |
| Mojo | Standalone InfoNCE/CSDP validation kernel | `mojo src/sc_neurocore/accel/mojo/kernels/ssl.mojo` |

`benchmarks/results/bench_contrastive_ssl.json` records the latest local,
non-isolated evidence. The Python public API measured 1000 deterministic calls
at 10117.413 InfoNCE calls/s and 26799.118 CSDP contrastive steps/s on the
current workstation. Rust, Julia, and Mojo validation checks all passed in that
same run. The artifact is a local regression record, not an isolated production
benchmark claim.

**Reference:** Ororbia 2024, Science Advances.

See [Tutorial 80: Contrastive SSL](../tutorials/80_contrastive.md).

::: sc_neurocore.contrastive.ssl
    options:
      show_root_heading: true
      members:
        - SpikeContrastiveLoss
        - CSDPRule

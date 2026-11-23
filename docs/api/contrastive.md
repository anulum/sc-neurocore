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

## Execution Lanes

`sc_neurocore.accel.contrastive_ssl` runs the InfoNCE loss, one CSDP phase and
the goodness score on a chosen lane: `backend="python"` (the original above and
the floor), `"julia"` (`accel/julia/contrastive/ssl.jl`) or `"mojo"`
(`accel/mojo/contrastive_ssl`). Every input is validated with the original's
own checks before any lane runs, so all lanes refuse the same inputs.
`backend="auto"` follows the measured order for kernel `contrastive_ssl` on the
host; without a committed measurement it is Python alone.

```python
from sc_neurocore.accel import contrastive_ssl

loss = contrastive_ssl.infonce_loss(view_a, view_b, temperature=0.5, backend="mojo")
weights = contrastive_ssl.csdp_update(weights, pre, post, positive=True, backend="julia")
```

Parity bounds against Python, held by `tests/test_contrastive_ssl_julia_parity.py`
and `tests/test_contrastive_ssl_mojo_parity.py`:

| Kernel | Julia | Mojo |
|--------|-------|------|
| InfoNCE loss | relative `1e-9` | relative `1e-9` |
| CSDP phase | exact | `4·eps·(abs(W) + lr·abs(post⊗pre) + decay·abs(W))` per element (fused multiply-adds) |
| Goodness | relative `1e-12` | relative `1e-12` |

## Validation and Benchmark Evidence

The public Python API is covered by the module-specific
`tests/test_contrastive.py` surface. The maintained polyglot mirrors are:

| Surface | Scope | Local check |
|---------|-------|-------------|
| Python | Public InfoNCE and CSDP contracts, validation guards, deterministic algebra | `pytest tests/test_contrastive.py` |
| Julia lane | Dispatcher lane; InfoNCE, CSDP and goodness parity | `pytest tests/test_contrastive_ssl_julia_parity.py` |
| Mojo lane | Dispatcher lane (C ABI); InfoNCE, CSDP and goodness parity | `pytest tests/test_contrastive_ssl_mojo_parity.py` |
| Rust | Standalone safety mirror for row-wise InfoNCE and CSDP matrix updates; not a dispatcher lane | `rustc --edition=2021 --test src/sc_neurocore/accel/rust/safety/ssl.rs` |
| Julia | Row-wise InfoNCE and CSDP validation of `ssl.jl` | `julia --startup-file=no --history-file=no ... validate_ssl()` |
| Mojo | Standalone InfoNCE/CSDP validation kernel | `mojo src/sc_neurocore/accel/mojo/kernels/ssl.mojo` |

`benchmarks/results/bench_contrastive_ssl.json` records the latest local,
non-isolated evidence, taken on 11th Gen Intel(R) Core(TM) i5-11600K @ 3.90GHz (CPU 0, load average
6.61 6.58 5.66). The Python public API measured 1000 deterministic calls
at 14561.756 InfoNCE calls/s and 47018.991 CSDP contrastive steps/s.
Through the dispatcher, on the same 32 × 64 views and 2 × 2 rule (calls/s):

| Lane | InfoNCE | CSDP step (two phases) | Goodness |
|------|--------:|-----------------------:|---------:|
| julia | 10819.629 | 11071.250 | 4053.938 |
| mojo | 11456.027 | 30230.221 | 93664.119 |
| python | 18742.408 | 24815.532 | 97229.617 |

Rust, Julia, and Mojo validation checks all passed in that same run. The
artifact is a local regression record, not an isolated production benchmark
claim; timings on a loaded workstation vary between runs.

**Reference:** Ororbia 2024, Science Advances.

See [Tutorial 80: Contrastive SSL](../tutorials/80_contrastive.md).

::: sc_neurocore.contrastive.ssl
    options:
      show_root_heading: true
      members:
        - SpikeContrastiveLoss
        - CSDPRule

::: sc_neurocore.accel.contrastive_ssl
    options:
      show_root_heading: true
      members:
        - infonce_loss
        - csdp_update
        - goodness
        - backend_available
        - auto_backend

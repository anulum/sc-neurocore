# Differential Privacy — Spike-Level DP

Spike-level differential privacy: add privacy noise at the spike domain instead of the gradient domain. Exploits the binary nature of spikes for more natural DP mechanisms.

## Why Spike-Level DP?

Standard DP-SGD adds Gaussian noise to gradients (continuous, high-dimensional). For SNNs, spikes are already binary — we can use mechanisms designed for binary data:

| Mechanism | How It Works | Privacy Cost |
|-----------|-------------|-------------|
| Randomized Response | Flip each bit with probability `p = 1/(1+e^ε)` | ε per bit |
| Poisson Subsampling | Keep each spike with probability `q = e^ε/(1+e^ε)` | ε per step |

## Components

- **`SpikeLevelDP`** — Main DP mechanism.

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `epsilon` | 1.0 | Per-step privacy budget |
| `mechanism` | "randomized_response" | DP mechanism |

Methods: `privatize(spikes)` — apply DP noise to a spike tensor.

- **`PrivacyAccountant`** — Track cumulative privacy budget.

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `target_epsilon` | 1.0 | Total privacy budget |
| `target_delta` | 1e-5 | Failure probability |

Properties: `spent_epsilon`, `remaining_epsilon`, `budget_exhausted`. Methods: `record_step(step_epsilon)`, `summary()`.

- **`MembershipAudit`** — Audit SNN for membership inference vulnerability. Compares model confidence on training vs non-training samples. Returns accuracy (0.5 = no leakage, 1.0 = full leak), `vulnerable` flag if accuracy > 0.6.

## Usage

```python
from sc_neurocore.privacy.dp_snn import SpikeLevelDP, PrivacyAccountant, MembershipAudit
import numpy as np

# Apply DP to spike outputs
dp = SpikeLevelDP(epsilon=1.0, mechanism="randomized_response")
spikes = np.random.randint(0, 2, (100, 64)).astype(np.int8)
private_spikes = dp.privatize(spikes)

# Track privacy budget
accountant = PrivacyAccountant(target_epsilon=10.0)
for step in range(100):
    accountant.record_step(dp.per_step_epsilon)
    if accountant.budget_exhausted:
        print(f"Budget exhausted at step {step}")
        break
print(accountant.summary())

# Membership inference audit
def model_fn(x):
    return np.random.randn(10)  # your model here

auditor = MembershipAudit(run_fn=model_fn)
result = auditor.audit(member_samples, non_member_samples)
print(f"MI accuracy: {result['accuracy']:.2f}, vulnerable: {result['vulnerable']}")
```

See [Tutorial 62: Differential Privacy](../tutorials/62_privacy.md).

::: sc_neurocore.privacy
    options:
      show_root_heading: true

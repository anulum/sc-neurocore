# Tutorial 62: Differential Privacy for SNNs

Add mathematical privacy guarantees to SNN training and inference.
Spike-level differential privacy ensures that individual training
samples cannot be identified from the trained model's outputs — even
by an adversary with full access to the model weights.

## Why Privacy for SNNs

SNNs deployed in medical (BCI, EEG analysis) and biometric (gait,
voice) applications process sensitive personal data. Without privacy
protection, trained weights can leak information about individual
training subjects (membership inference attacks).

| Threat | Without DP | With DP |
|--------|:----------:|:-------:|
| Membership inference | Vulnerable | Protected (ε-bounded) |
| Model inversion | Possible | Bounded noise prevents |
| Training data extraction | Risk of exact memorisation | Provably impossible (above ε) |

## Spike-Level DP

Randomised response at the spike level — each spike is independently
flipped with probability determined by ε:

```python
import numpy as np
from sc_neurocore.privacy import SpikeLevelDP, PrivacyAccountant

rng = np.random.default_rng(42)

dp = SpikeLevelDP(
    epsilon=1.0,                    # privacy budget (lower = more private)
    mechanism="randomized_response", # spike-level random flip
)

raw_spikes = (rng.random((100, 64)) < 0.1).astype(np.int8)
private_spikes = dp.privatize(raw_spikes)

print(f"Original spikes: {raw_spikes.sum()}")
print(f"Private spikes:  {private_spikes.sum()}")
print(f"Changed:         {(raw_spikes != private_spikes).sum()}")
```

### How Randomised Response Works

For each spike position, with probability p = 1/(1 + e^ε):
- If spike=1: flip to 0 (suppress a real spike)
- If spike=0: flip to 1 (inject a false spike)

At ε=1.0, ~27% of values are randomised. At ε=0.1, ~47% are
randomised (very private but noisy). At ε=10.0, ~0.005% are
randomised (barely private but high utility).

## Privacy Budget Tracking

Each training step consumes privacy budget. The accountant tracks
cumulative ε and warns when the budget is exhausted:

```python
accountant = PrivacyAccountant(target_epsilon=10.0)

for epoch in range(100):
    # Each epoch uses the DP mechanism
    accountant.record_step(dp.per_step_epsilon)

    if accountant.budget_exhausted:
        print(f"Budget exhausted at epoch {epoch}")
        break

print(accountant.summary())
# Total ε consumed: 10.0 / 10.0
# Steps taken: 10
# ε per step: 1.0
# Budget status: EXHAUSTED
```

## Membership Inference Audit

Test whether your model leaks membership information:

```python
from sc_neurocore.privacy import MembershipAudit

def my_model(x):
    """Your SNN forward pass."""
    return np.random.rand(10)  # replace with actual inference

audit = MembershipAudit(run_fn=my_model)

training_samples = rng.standard_normal((100, 784)).astype(np.float32)
holdout_samples = rng.standard_normal((100, 784)).astype(np.float32)

result = audit.audit(training_samples, holdout_samples)
print(f"Inference accuracy: {result['accuracy']:.2%}")
print(f"Vulnerable: {result['vulnerable']}")
# accuracy > 55% → model leaks membership information
# accuracy ≈ 50% → model is safe (can't distinguish train from holdout)
```

## Choosing ε

| ε Value | Privacy Level | Utility Impact | Use Case |
|---------|:------------:|:--------------:|----------|
| 0.1 | Very strong | High noise (5-15% accuracy drop) | Medical data |
| 1.0 | Strong | Moderate noise (1-5% drop) | Biometric data |
| 10.0 | Moderate | Minimal noise (<1% drop) | General ML |
| ∞ | None | No noise | Non-sensitive data |

## References

- Dwork & Roth (2014). "The Algorithmic Foundations of Differential
  Privacy." Foundations and Trends in Theoretical Computer Science.
- Kim et al. (2023). "Differentially Private Spiking Neural Networks."
  NeurIPS 2023 Workshop on Privacy-Preserving ML.

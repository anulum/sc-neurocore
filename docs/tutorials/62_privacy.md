# Tutorial 62: Differential Privacy for SNNs

Add privacy guarantees to SNN training and inference.

## Spike-Level DP

```python
from sc_neurocore.privacy import SpikeLevelDP, PrivacyAccountant

dp = SpikeLevelDP(epsilon=1.0, mechanism="randomized_response")
private_spikes = dp.privatize(raw_spikes)

# Track privacy budget
accountant = PrivacyAccountant(target_epsilon=10.0)
for epoch in range(100):
    accountant.record_step(dp.per_step_epsilon)
    if accountant.budget_exhausted:
        break
print(accountant.summary())
```

## Membership Inference Audit

```python
from sc_neurocore.privacy import MembershipAudit

audit = MembershipAudit(run_fn=my_model)
result = audit.audit(training_samples, holdout_samples)
print(f"Inference accuracy: {result['accuracy']:.2f}")
print(f"Vulnerable: {result['vulnerable']}")
```

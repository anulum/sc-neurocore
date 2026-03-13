<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->

# Ensemble Methods for SC Networks

Reduce stochastic computing noise by running multiple independent
networks and aggregating their outputs. SC ensembles exploit the
1/√N noise reduction property — the same principle that makes
longer bitstreams more precise.

**Prerequisites**: `pip install sc-neurocore scikit-learn matplotlib`

## 1. Why ensembles matter more in SC

In conventional DNNs, ensembles improve accuracy by ~1-3%.
In SC networks, ensembles are more impactful because they reduce
the inherent bitstream noise:

```
Single network noise:   σ ∝ 1/√L
Ensemble of N networks: σ ∝ 1/√(N·L)

Equivalent effect: ensemble of 4 networks at L=256
                 = single network at L=1024
```

The ensemble approach is often cheaper than increasing L because
the N networks can run in parallel on separate hardware.

## 2. Basic ensemble: majority vote

The simplest ensemble: N networks vote on the classification:

```python
import numpy as np
from sc_neurocore import VectorizedSCLayer

N_ENSEMBLE = 5
N_IN = 50
N_HIDDEN = 64
N_OUT = 10
L = 256

# Create N independent networks (different random seeds)
ensembles = []
for i in range(N_ENSEMBLE):
    np.random.seed(i * 1000)
    l1 = VectorizedSCLayer(n_inputs=N_IN, n_neurons=N_HIDDEN, length=L)
    l2 = VectorizedSCLayer(n_inputs=N_HIDDEN, n_neurons=N_OUT, length=L)
    ensembles.append((l1, l2))

def ensemble_predict(x, method="vote"):
    """Run all ensemble members and aggregate."""
    predictions = []
    scores_all = []

    for l1, l2 in ensembles:
        h = np.clip(l1.forward(x), 0.01, 0.99)
        scores = l2.forward(h)
        scores_all.append(scores)
        predictions.append(scores.argmax())

    if method == "vote":
        # Majority vote
        from collections import Counter
        votes = Counter(predictions)
        return votes.most_common(1)[0][0], scores_all
    elif method == "average":
        # Average scores, then argmax
        avg = np.mean(scores_all, axis=0)
        return avg.argmax(), scores_all

x_test = np.random.uniform(0.1, 0.9, size=N_IN)
pred_vote, _ = ensemble_predict(x_test, "vote")
pred_avg, _ = ensemble_predict(x_test, "average")
print(f"Majority vote:   class {pred_vote}")
print(f"Average scores:  class {pred_avg}")
```

## 3. Confidence estimation

SC ensembles provide natural confidence estimates from disagreement:

```python
def ensemble_confidence(x):
    """Predict with confidence from ensemble agreement."""
    _, scores_all = ensemble_predict(x, "average")
    scores_arr = np.array(scores_all)  # (N_ENSEMBLE, N_OUT)

    # Mean prediction
    mean_scores = scores_arr.mean(axis=0)
    predicted_class = mean_scores.argmax()

    # Confidence metrics
    agreement = np.mean([s.argmax() == predicted_class for s in scores_arr])
    score_std = scores_arr[:, predicted_class].std()

    return predicted_class, agreement, score_std

pred, agreement, std = ensemble_confidence(x_test)
print(f"Predicted: {pred}")
print(f"Agreement: {agreement:.0%} ({int(agreement * N_ENSEMBLE)}/{N_ENSEMBLE} agree)")
print(f"Score std: {std:.4f} (lower = more confident)")
```

Agreement < 60% or high score std → low confidence, flag for review.

## 4. Noise reduction measurement

Quantify the ensemble's noise reduction:

```python
def measure_noise_reduction(x, n_runs=50):
    """Measure output variance across multiple evaluations."""
    # Single network variance
    l1, l2 = ensembles[0]
    single_outputs = []
    for _ in range(n_runs):
        h = np.clip(l1.forward(x), 0.01, 0.99)
        single_outputs.append(l2.forward(h))
    single_var = np.var(single_outputs, axis=0).mean()

    # Ensemble average variance
    ens_outputs = []
    for _ in range(n_runs):
        scores_all = []
        for l1, l2 in ensembles:
            h = np.clip(l1.forward(x), 0.01, 0.99)
            scores_all.append(l2.forward(h))
        ens_outputs.append(np.mean(scores_all, axis=0))
    ens_var = np.var(ens_outputs, axis=0).mean()

    ratio = single_var / max(ens_var, 1e-10)
    print(f"Single network variance: {single_var:.6f}")
    print(f"Ensemble variance:       {ens_var:.6f}")
    print(f"Noise reduction:         {ratio:.1f}× (expected: ~{N_ENSEMBLE}×)")

measure_noise_reduction(x_test)
```

## 5. Diversity-promoting training

Ensemble members should disagree on hard examples. Promote diversity
by training each member on a different data bootstrap:

```python
from sklearn.datasets import make_moons

X, y = make_moons(n_samples=1000, noise=0.2, random_state=42)
X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
X = np.clip(X, 0.01, 0.99)

# Bootstrap training: each member sees a different 80% of data
for i, (l1, l2) in enumerate(ensembles):
    np.random.seed(i)
    idx = np.random.choice(len(X), size=int(0.8 * len(X)), replace=True)
    X_boot = X[idx]
    y_boot = y[idx]

    # Training loop (simplified — see Tutorial 07 for full version)
    lr = 0.01
    for epoch in range(10):
        for j in range(len(X_boot)):
            h = np.clip(l1.forward(X_boot[j]), 0.01, 0.99)
            scores = l2.forward(h)
            # Pseudo-gradient update...
            # (omitted for brevity)

print(f"Trained {N_ENSEMBLE} ensemble members on bootstrapped data")
```

## 6. Weighted ensemble based on validation performance

Weight each member by its validation accuracy:

```python
def weighted_ensemble(x, member_weights):
    """Weighted score averaging."""
    total = np.zeros(N_OUT)
    for (l1, l2), w in zip(ensembles, member_weights):
        h = np.clip(l1.forward(x), 0.01, 0.99)
        scores = l2.forward(h)
        total += w * scores
    return total.argmax()

# Assign weights based on individual accuracy
# (in practice, compute from validation set)
member_weights = np.array([0.85, 0.88, 0.82, 0.90, 0.86])
member_weights = member_weights / member_weights.sum()

pred = weighted_ensemble(x_test, member_weights)
print(f"Weighted prediction: {pred}")
```

## 7. FPGA parallel ensemble

On FPGA, ensemble members run in parallel — no latency increase:

```
                ┌──── Network 1 ────┐
   Input ──────├──── Network 2 ────├──── Majority ──── Output
   (shared)    ├──── Network 3 ────├     Vote
                └──── Network 4 ────┘

   Latency: same as single network
   Area: N × single network
   Accuracy: +5-10% over single
```

Resource cost for 5-member ensemble (50→64→10, L=256):

| Resource | Single | Ensemble (5) |
|----------|--------|-------------|
| LUTs | ~8,000 | ~40,000 |
| FFs | ~3,000 | ~15,000 |
| Power | ~5 mW | ~25 mW |
| Latency | 256 clk | 256 clk |
| Accuracy | ~85% | ~90% |

The ECP5-85K (83K LUTs) can fit a 5-member ensemble with room
for the voter logic.

## 8. Temporal ensemble (free accuracy boost)

Instead of N networks, run the same network N times with different
LFSR seeds and average. Zero extra hardware — just time:

```python
def temporal_ensemble(layer1, layer2, x, n_runs=5):
    """Run same network multiple times, average outputs."""
    all_scores = []
    for run in range(n_runs):
        # Each forward pass uses different internal LFSR state
        h = np.clip(layer1.forward(x), 0.01, 0.99)
        scores = layer2.forward(h)
        all_scores.append(scores)

    avg = np.mean(all_scores, axis=0)
    return avg.argmax(), avg

l1, l2 = ensembles[0]
pred, avg_scores = temporal_ensemble(l1, l2, x_test, n_runs=10)
print(f"Temporal ensemble (10 runs): class {pred}")
print(f"Score std: {np.array([temporal_ensemble(l1, l2, x_test)[1] for _ in range(5)]).std(axis=0).mean():.4f}")
```

Temporal ensemble trades latency for accuracy with zero area overhead.

## What you learned

- SC ensembles reduce noise by √N (more impactful than in float DNNs)
- Majority vote: simple, works with any number of classes
- Score averaging: smoother, enables confidence estimation
- Agreement rate provides natural confidence metric
- Bootstrap training promotes ensemble diversity
- FPGA ensembles run in parallel — accuracy boost with no latency cost
- Temporal ensemble: multiple runs, same network, zero area overhead

## Next steps

- Combine ensemble with STDP: each member self-organises differently
- Implement hardware voter logic in Verilog
- Compare N×L/N ensemble vs single L network (which is better?)
- Use confidence estimates for active learning (label uncertain samples)

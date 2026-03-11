<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->

# Stochastic Computing Fundamentals

Stochastic computing (SC) encodes real-valued quantities as the probability
of a 1-bit in a random bitstream. Arithmetic reduces to single-gate logic:
AND computes multiplication, OR computes (a capped form of) addition. The
tradeoff is precision — a bitstream of length *L* carries at most log₂(*L*)
bits of information, and estimation variance scales as 1/*L*.

This tutorial covers the mathematical foundations and the SC-NeuroCore API
for encoding, arithmetic, and variance analysis.

**Prerequisites:** Python 3.10+, `pip install sc-neurocore numpy`

## 1. Unipolar encoding

A scalar *x* ∈ [0, 1] is represented by a bitstream **b** = (b₁, …, b_L)
where each bᵢ ∈ {0, 1} and P(bᵢ = 1) = *x*. The proportion of ones
converges to *x* as *L* → ∞.

```python
import numpy as np
from sc_neurocore import generate_bernoulli_bitstream, bitstream_to_probability

p = 0.7
L = 1024
bs = generate_bernoulli_bitstream(p, length=L)

print(f"dtype={bs.dtype}, shape={bs.shape}")       # uint8, (1024,)
print(f"first 16 bits: {bs[:16]}")                  # mix of 0s and 1s
print(f"estimated p = {bitstream_to_probability(bs):.4f}")  # ≈ 0.70
```

`generate_bernoulli_bitstream` draws each bit independently from
Bernoulli(*p*) using NumPy's PCG64 generator (wrapped by `RNG`).

## 2. Sobol encoding: faster convergence

Bernoulli bitstreams have estimation error O(1/√*L*). Sobol (quasi-random)
sequences are low-discrepancy: for the same *L*, the error is O(1/*L*) —
a quadratic improvement.

```python
from sc_neurocore import generate_sobol_bitstream

bs_sobol = generate_sobol_bitstream(0.7, length=1024, seed=42)
print(f"Sobol p̂ = {bitstream_to_probability(bs_sobol):.4f}")
```

Under the hood, `scipy.stats.qmc.Sobol` generates *L* samples in [0,1)
and thresholds at *p*. Power-of-two lengths preserve the equidistribution
properties of Sobol sequences.

## 3. BitstreamEncoder: mapping arbitrary ranges

Physical quantities (membrane voltages, synaptic currents) live on [x_min,
x_max], not [0, 1]. `BitstreamEncoder` applies the linear map
p = (x − x_min)/(x_max − x_min) before encoding.

```python
from sc_neurocore import BitstreamEncoder, bitstream_to_probability

enc = BitstreamEncoder(x_min=0.0, x_max=5.0, length=2048, mode="bernoulli", seed=7)
x = 3.5
bs = enc.encode(x)                     # p = 3.5/5 = 0.7
x_hat = enc.decode(bs)                 # inverse map
print(f"original={x}, recovered={x_hat:.4f}")

enc_s = BitstreamEncoder(x_min=0.0, x_max=5.0, length=2048, mode="sobol", seed=7)
x_hat_s = enc_s.decode(enc_s.encode(x))
print(f"Sobol recovered={x_hat_s:.4f}")
```

## 4. Bitstream arithmetic

### Multiplication via AND

For two independent bitstreams **a** ~ Bernoulli(*p*) and **b** ~
Bernoulli(*q*):

> P(aᵢ AND bᵢ = 1) = P(aᵢ = 1) · P(bᵢ = 1) = *pq*

Independence is critical — correlated streams violate this identity.

```python
p, q = 0.6, 0.8
L = 4096
a = generate_bernoulli_bitstream(p, L)
b = generate_bernoulli_bitstream(q, L)

product = np.bitwise_and(a, b)
print(f"p·q = {p*q:.2f}, AND estimate = {bitstream_to_probability(product):.4f}")
```

### Addition via OR (saturating)

OR implements clamped addition. For independent streams:

> P(aᵢ OR bᵢ = 1) = p + q − pq

This is *not* p + q. OR saturates at 1 and is exact only when pq ≈ 0.
For small operands (common in neural spike trains), the approximation
p + q − pq ≈ p + q holds.

```python
p, q = 0.1, 0.15
a = generate_bernoulli_bitstream(p, L)
b = generate_bernoulli_bitstream(q, L)

or_result = np.bitwise_or(a, b)
expected = p + q - p * q
print(f"p+q-pq = {expected:.4f}, OR estimate = {bitstream_to_probability(or_result):.4f}")
```

## 5. Variance analysis

The estimator p̂ = (1/L) Σ bᵢ is unbiased with variance:

> σ² = p(1 − p) / L

This is the Bernoulli sampling variance. The maximum σ² occurs at p = 0.5
(worst case: σ² = 1/(4L)). Doubling precision (halving σ) requires 4× the
bitstream length.

```python
p = 0.5
trials = 1000
for L in [64, 256, 1024, 4096]:
    estimates = [
        bitstream_to_probability(generate_bernoulli_bitstream(p, L))
        for _ in range(trials)
    ]
    empirical_var = np.var(estimates)
    theory_var = p * (1 - p) / L
    print(f"L={L:5d}  σ²_theory={theory_var:.6f}  σ²_empirical={empirical_var:.6f}")
```

Expected output confirms 1/L scaling:

```
L=   64  σ²_theory=0.003906  σ²_empirical=0.003900
L=  256  σ²_theory=0.000977  σ²_empirical=0.000980
L= 1024  σ²_theory=0.000244  σ²_empirical=0.000242
L= 4096  σ²_theory=0.000061  σ²_empirical=0.000061
```

Sobol sequences break the 1/√L barrier. Their effective variance scales
closer to 1/L², though the exact rate depends on the integrand's smoothness:

```python
for L in [64, 256, 1024, 4096]:
    estimates = [
        bitstream_to_probability(generate_sobol_bitstream(p, L, seed=s))
        for s in range(trials)
    ]
    empirical_var = np.var(estimates)
    theory_var = p * (1 - p) / L
    print(f"L={L:5d}  σ²_bernoulli_theory={theory_var:.6f}  σ²_sobol={empirical_var:.8f}")
```

## 6. Multiplication error propagation

When multiplying two bitstreams encoding *p* and *q*, the output encodes
*pq* with combined variance. For independent Bernoulli streams:

> Var(p̂q̂) ≈ q²·Var(p̂) + p²·Var(q̂) = (p²q(1−q) + q²p(1−p)) / L

```python
p, q = 0.6, 0.8
L = 2048
trials = 2000

products = []
for _ in range(trials):
    a = generate_bernoulli_bitstream(p, L)
    b = generate_bernoulli_bitstream(q, L)
    products.append(bitstream_to_probability(np.bitwise_and(a, b)))

empirical_var = np.var(products)
# delta method: Var(pq) ≈ q²σ²_p + p²σ²_q
theory_var = (q**2 * p * (1 - p) + p**2 * q * (1 - q)) / L
print(f"Var(AND) theory={theory_var:.7f}, empirical={empirical_var:.7f}")
```

## What you learned

- SC encodes values as bit probabilities; decoding is a sample mean.
- Bernoulli encoding: σ² = p(1−p)/L. Sobol encoding converges faster.
- AND gate = multiplication (requires independent streams).
- OR gate = saturating addition: p + q − pq, not p + q.
- Precision costs are quadratic: halving error requires 4× stream length.
- `BitstreamEncoder` handles the [x_min, x_max] ↔ [0, 1] mapping.
- All code above runs with `pip install sc-neurocore` — no Rust engine,
  no hardware dependencies.

<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->

# Stochastic Computing Theory: A Deep Dive

The mathematical foundations of stochastic computing, from Gaines (1969)
to modern SC neural networks. This tutorial bridges the gap between
"it works" and "here's why it works."

**Prerequisites**: Basic probability theory, familiarity with Tutorial 01.

## 1. Bernoulli bitstreams as random variables

A stochastic bitstream of length L encoding probability p is a sequence
of i.i.d. Bernoulli(p) random variables:

```
B = (b_1, b_2, ..., b_L)  where b_i ~ Bernoulli(p)

P(b_i = 1) = p
P(b_i = 0) = 1 - p

Estimator: p̂ = (1/L) Σ b_i
E[p̂] = p                    (unbiased)
Var(p̂) = p(1-p)/L           (central limit theorem)
σ(p̂) = √(p(1-p)/L)         (standard error)
```

For p=0.5 (worst case), σ = 1/(2√L):

| L | σ | Effective bits |
|---|---|---------------|
| 64 | 0.063 | ~4 |
| 256 | 0.031 | ~5 |
| 1024 | 0.016 | ~6 |
| 4096 | 0.008 | ~7 |

Effective bits ≈ -log₂(σ). The 1/√L scaling means doubling precision
costs 4× the bitstream length — diminishing returns.

## 2. Multiplication via AND

For independent bitstreams A ~ Bernoulli(p_a) and B ~ Bernoulli(p_b):

```
C = A AND B

P(C = 1) = P(A=1) · P(B=1) = p_a · p_b
```

This is exact multiplication with zero error — as long as A and B
are independent. Independence is the fundamental requirement.

### Correlation destroys accuracy

If A and B share the same random source:
```
P(A AND B) = P(A=1, B=1) = p_a·p_b + ρ·√(p_a(1-p_a)·p_b(1-p_b))

where ρ is the Pearson correlation between the bitstreams.
```

For ρ=1 (same LFSR): P(A AND A) = p_a ≠ p_a² (unless p_a ∈ {0, 1}).
This is squaring, not multiplication. Useful for computing x² but
not for general weight × input.

```python
import numpy as np

L = 10000
p_a, p_b = 0.7, 0.4

# Independent
rng_a = np.random.RandomState(42)
rng_b = np.random.RandomState(137)
A = (rng_a.rand(L) < p_a).astype(int)
B = (rng_b.rand(L) < p_b).astype(int)
print(f"Independent: AND mean = {(A & B).mean():.4f} (expected {p_a*p_b:.4f})")

# Correlated (same RNG)
rng_shared = np.random.RandomState(42)
U = rng_shared.rand(L)
A_corr = (U < p_a).astype(int)
B_corr = (U < p_b).astype(int)
print(f"Correlated:  AND mean = {(A_corr & B_corr).mean():.4f} (expected {min(p_a, p_b):.4f})")
```

With shared randomness, AND(A, B) = min(p_a, p_b) — this is the
stochastic MIN operation, not multiplication.

## 3. Addition via MUX

A 2:1 multiplexer with select line S ~ Bernoulli(0.5):

```
C = S ? A : B

P(C=1) = P(S=1)·P(A=1) + P(S=0)·P(B=1) = 0.5·p_a + 0.5·p_b
```

This computes the scaled sum (p_a + p_b)/2. For N inputs with
random selection:

```
P(C=1) = (1/N) Σ p_i
```

Exact arithmetic average — no carry propagation, no overflow.

## 4. The SC noise floor

Every SC operation adds noise. For a chain of K operations:

```
Total variance ≈ Σ_k Var(output_k) / L_k
```

In a multi-layer network with layers of sizes N_1, N_2, ..., N_K:
```
Output σ ≈ √(K / L)  (rough approximation)
```

This is why deep SC networks (K > 5) need longer bitstreams or
ensemble averaging — noise compounds through layers.

```python
# Demonstrate noise compounding
from sc_neurocore import VectorizedSCLayer

def measure_layer_noise(n_layers, L, n_trials=100):
    layers = [VectorizedSCLayer(n_inputs=32, n_neurons=32, length=L)
              for _ in range(n_layers)]
    x = np.full(32, 0.5)  # constant input

    outputs = []
    for _ in range(n_trials):
        h = x.copy()
        for layer in layers:
            h = np.clip(layer.forward(h), 0.01, 0.99)
        outputs.append(h.mean())

    return np.std(outputs)

print("Noise vs depth (L=256):")
for depth in [1, 2, 3, 5, 8]:
    noise = measure_layer_noise(depth, L=256)
    print(f"  {depth} layers: σ = {noise:.4f}")
```

## 5. XNOR for bipolar encoding

Unipolar SC represents [0, 1]. For signed values [-1, 1], use
bipolar encoding:

```
Value x ∈ [-1, 1]
P(bit=1) = (x + 1) / 2

Multiplication: XNOR(A, B)
P(XNOR = 1) = P(A=1)·P(B=1) + P(A=0)·P(B=0)
             = p_a·p_b + (1-p_a)(1-p_b)
             = 2·p_a·p_b - p_a - p_b + 1
```

Converting back to bipolar values:
```
x_a = 2p_a - 1,  x_b = 2p_b - 1
XNOR encodes: x_a · x_b
```

XNOR is an exact signed multiplier in bipolar SC.

## 6. Stochastic number precision theory

The Fisher information of a Bernoulli(p) bitstream of length L:

```
I(p) = L / (p(1-p))
```

Cramér-Rao lower bound on estimator variance:
```
Var(p̂) ≥ 1/I(p) = p(1-p)/L
```

The sample mean achieves this bound — it is the most efficient
estimator. No encoding scheme can do better than 1/√L precision
for L random bits.

### Comparison with deterministic binary

A deterministic B-bit binary number has precision 2^(-B).
An SC bitstream of length L = 2^B has precision ~1/√(2^B) = 2^(-B/2).

SC is exponentially less precise per bit. But each bit requires
1 gate to process, while binary requires O(2^B) gates. The total
"precision × area" product is comparable.

## 7. Sobol sequences for better precision

LFSR bitstreams are not truly random — they are pseudo-random with
known correlation structure. Low-discrepancy sequences (Sobol, Halton)
can improve SC precision beyond 1/√L:

```python
# Standard LFSR encoding
from sc_neurocore import BitstreamEncoder

enc_lfsr = BitstreamEncoder(length=256, seed=42)
bits_lfsr = enc_lfsr.encode(0.3)
error_lfsr = abs(bits_lfsr.mean() - 0.3)

# Sobol-like deterministic encoding (for comparison)
# Generate bits by comparing p against evenly spaced thresholds
thresholds = np.linspace(0, 1, 256, endpoint=False)
np.random.shuffle(thresholds)  # shuffle to avoid systematic patterns
bits_sobol = (thresholds < 0.3).astype(np.uint8)
error_sobol = abs(bits_sobol.mean() - 0.3)

print(f"LFSR error:  {error_lfsr:.6f}")
print(f"Sobol error: {error_sobol:.6f}")
```

Sobol encoding achieves O(log(L)/L) error vs O(1/√L) for random —
but loses the independence property needed for AND multiplication.
Use Sobol only when computing single values, not products.

## 8. Error propagation in SC neural networks

For a two-layer SC network with layers W₁ (n×m) and W₂ (m×k):

```
h = SC_forward(W₁, x)     error: σ_h ≈ √(n/(L·m))
y = SC_forward(W₂, h)     error: σ_y ≈ √(m/(L·k)) + propagated σ_h

Total output error: σ_y ≈ √(1/L) · (√(n/m) + √(m/k))
```

This predicts:
- Wider layers (larger m) reduce noise (more averaging in the MUX tree)
- Output error scales as 1/√L (confirming the fundamental limit)
- Deeper networks accumulate errors additively

## 9. Information-theoretic capacity

The mutual information between SC input and output:

```
I(X; Y) ≤ L · C_BSC(p_e)

where C_BSC(p_e) = 1 - H(p_e) is the binary symmetric channel capacity
and p_e is the effective bit error rate.
```

For L=256 and a well-designed SC circuit (p_e ≈ 0.01):
```
I(X; Y) ≤ 256 · (1 - H(0.01)) ≈ 256 · 0.92 ≈ 235 bits
```

This is the theoretical maximum information an SC link of length 256
can carry per encoding — about 29 bytes. A single float32 carries 32
bits. So an SC encoding of L=256 carries ~7× more information than a
single float, but spread across 256 clock cycles.

## What you learned

- Bernoulli bitstreams are unbiased estimators with σ ∝ 1/√L
- AND gate = exact multiplication (requires independent inputs)
- MUX = exact scaled addition (no overflow possible)
- Correlated bitstreams break multiplication (AND becomes MIN)
- XNOR implements signed (bipolar) multiplication
- Noise compounds through layers: keep networks wide and shallow
- Sobol encoding can beat 1/√L for single values but breaks AND
- SC trades exponential area savings for polynomial precision loss

## Further reading

- Gaines, B.R. (1969). "Stochastic Computing Systems." Advances in
  Information Systems Science, Vol. 2.
- Alaghi & Hayes (2013). "Survey of Stochastic Computing."
  ACM TODAES, 18(2).
- Tutorial 01: SC Fundamentals (practical introduction)
- Tutorial 16: Debugging SC Networks (applied noise diagnosis)

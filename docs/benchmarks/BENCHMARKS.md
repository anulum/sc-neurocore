# sc-neurocore Benchmarks

This file records performance and accuracy measurements for sc-neurocore. Use it to track changes over time and to compare hardware and software runs.

---

## 1. Benchmark philosophy

Stochastic computing results are statistical. Benchmarks should capture both performance and output distribution characteristics. For each benchmark run, record:

- Hardware or system configuration
- Bitstream length
- Network size and layer types
- Runtime per step or per batch
- Output mean and variance for key metrics
- Random seed strategy

---

## 2. Recommended benchmark sets

### 2.1 Core neuron throughput
- Model: StochasticLIFNeuron
- Inputs: constant current
- Outputs: spikes per second
- Purpose: baseline throughput for single neuron dynamics

### 2.2 Dense layer scaling
- Model: SCDenseLayer
- Inputs: fixed vector
- Outputs: spike matrix shape, firing rates
- Purpose: measure cost of dense layer simulation

### 2.3 Vectorized layer speed
- Model: VectorizedSCLayer
- Inputs: random vector
- Outputs: mean current
- Purpose: measure packed bitstream performance

### 2.4 Memristive defect impact
- Model: MemristiveDenseLayer
- Inputs: fixed vector
- Outputs: output mean and variance
- Purpose: quantify hardware defect effects

### 2.5 Transformer block demo
- Model: StochasticTransformerBlock
- Inputs: single token
- Outputs: output distribution
- Purpose: validate higher order integration overhead

---

## 3. Benchmark log template

Copy and fill the following template for each run:

```
Date:
Environment:
  OS:
  CPU:
  RAM:
  Python:
  numpy:
  Notes:

Benchmark:
  Name:
  Model:
  Inputs:
  Bitstream length:
  Layers / parameters:
  RNG seed strategy:

Results:
  Runtime:
  Throughput:
  Output mean:
  Output variance:
  Notes:
```

---

## 4. Example entries

### 4.1 Example: Vectorized layer CPU

Date: 2026-01-27
Environment:
  OS: Windows 11
  CPU: (fill)
  RAM: (fill)
  Python: 3.11
  numpy: 1.26

Benchmark:
  Name: Vectorized layer baseline
  Model: VectorizedSCLayer
  Inputs: 64 values, uniform 0.5
  Bitstream length: 1024
  Layers / parameters: n_inputs=64, n_neurons=256
  RNG seed strategy: numpy seed 42

Results:
  Runtime: (fill)
  Throughput: (fill)
  Output mean: (fill)
  Output variance: (fill)
  Notes: baseline for packed bitstream operations

---

## 5. Notes

- Keep old entries. Trends matter more than isolated numbers.
- When comparing versions, record the git commit or tag.
- Use consistent input distributions for fair comparisons.


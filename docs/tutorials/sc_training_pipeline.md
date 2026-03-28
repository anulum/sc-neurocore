# SC Bitstream MNIST Pipeline

First attempt at stochastic computing inference through an SNN trained
on MNIST.

## Pipeline

1. Train float SpikingNet(784->128->10) on MNIST: **96.2%** accuracy
2. Export weights via `to_sc_weights()`: normalise to [0,1]
3. SC inference: Bernoulli bitstream encoding, AND multiplication, popcount

## Results (Kaggle run 2026-03-28)

| Bitstream Length L | SC Accuracy | Float Accuracy | Drop |
|:------------------:|:-----------:|:--------------:|:----:|
| 64 | 9.0% | 96.2% | 87.2% |
| 128 | 11.2% | 96.2% | 85.0% |
| 256 | 8.2% | 96.2% | 88.0% |
| 512 | 10.8% | 96.2% | 85.4% |
| 1024 | 10.6% | 96.2% | 85.6% |

## Analysis: Why SC Inference Failed

SC accuracy is at random chance (~10% for 10 classes). The naive
float-to-SC conversion does not preserve discriminative power. Causes:

1. **Weight normalisation destroys relative magnitudes**: `to_sc_weights()`
   maps all weights to [0,1] per layer. This removes the sign information
   and relative scale between layers that the float network relies on.

2. **No bipolar SC representation**: Standard SC uses unipolar [0,1]
   probabilities with AND multiplication. The trained SNN has both positive
   and negative weights. Bipolar SC (using XNOR for multiplication) is
   needed but not yet implemented.

3. **Layer-to-layer propagation**: SC dot product output is a probability.
   Feeding this into the next layer's input encoding requires calibration
   of the probability-to-current mapping. The current implementation uses
   a fixed [y_min, y_max] range that doesn't match what the LIF expects.

4. **No SC-aware training**: The float SNN was trained without quantisation
   or SC-awareness. QAT (Task 2.4) constrains weights during training to
   be SC-compatible.

## Path Forward

- Implement bipolar SC (XNOR multiplication for signed weights)
- Add SC-aware quantisation during training (QAT + SC constraint)
- Calibrate inter-layer probability mapping
- Consider SC-native training (gradient through bitstream operations)

## Bipolar SC Results (Fix 1 + Fix 3, Kaggle 2026-03-28)

After implementing bipolar XNOR multiplication and per-layer calibration:

| Bitstream Length L | Bipolar Accuracy | Unipolar Accuracy | Float |
|:------------------:|:----------------:|:-----------------:|:-----:|
| 64 | 17.4% | 9.0% | 96.2% |
| 128 | 24.0% | 11.2% | 96.2% |
| 256 | 25.0% | 8.2% | 96.2% |
| 512 | 33.6% | 10.8% | 96.2% |
| 1024 | **35.6%** | 10.6% | 96.2% |

Bipolar XNOR + calibration improved from random chance (10%) to 35.6%.
Accuracy increases with L (expected SC convergence). Remaining gap
is from layer calibration not fully matching float distributions.

## SC-Aware Training Results (Fix 2, Kaggle 2026-03-28)

SC-aware training injects bitstream noise during float training,
making the model robust to SC variance.

| Model | Float | SC L=256 | SC L=512 | SC L=1024 |
|-------|:-----:|:--------:|:--------:|:---------:|
| Standard SNN | 96.6% | 24.5% | 25.0% | 31.0% |
| SC-Aware L=256 | 96.8% | 25.5% | 32.5% | 34.0% |
| SC-Aware L=1024 | 96.6% | 27.5% | 34.5% | **40.5%** |

SC-aware training improves SC inference by **9.5 percentage points**
(31.0% -> 40.5% at L=1024) without degrading float accuracy.

## SC Pipeline Progression

| Method | SC Accuracy (L=1024) |
|--------|:-------------------:|
| Unipolar AND (naive) | ~10% (random) |
| Bipolar XNOR | 35.6% |
| Bipolar + SC-aware training | **40.5%** |

## Honest Assessment

- 40.5% is well above random (10%) but well below float (96.6%)
- Remaining gap from calibration compression + single-pass inference
- End-to-end SC training (Fix 4) is the path to 80%+
- Literature SC accuracy on MNIST with careful design: 85-95%

## Files

- `benchmarks/results/sc_mnist_results.json` -- unipolar
- `benchmarks/results/sc_mnist_bipolar_results.json` -- bipolar
- `benchmarks/results/sc_aware_results.json` -- SC-aware training
- `notebooks/sc_aware_standalone_kaggle.py` -- standalone Kaggle script

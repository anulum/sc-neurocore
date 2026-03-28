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

## Honest Assessment

The first-ever SC SNN inference attempt on MNIST failed at the
weight-conversion stage. This is consistent with the literature:
naive post-training conversion to SC typically fails; SC-aware training
or careful calibration is required. The float training pipeline
(96.2%) and SC infrastructure (bitstream generation, AND multiplication,
popcount) both work correctly.

## Files

- `benchmarks/results/sc_mnist_results.json` -- measured data
- `notebooks/sc_mnist_pipeline_kaggle.py` -- Kaggle script

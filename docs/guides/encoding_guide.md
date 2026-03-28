# Temporal Encoding Comparison Guide

SC-NeuroCore provides 7 spike encoding schemes. This guide shows their
measured accuracy and spike efficiency on MNIST.

## Encodings

| Encoding | Principle | Spikes/sample | MNIST Accuracy |
|----------|-----------|:-------------:|:--------------:|
| repeat_binary | Threshold > 0.5, repeat T times | 2,429 | **91.9%** |
| direct | Float values repeated T times | 2,396 | **91.2%** |
| rate | Bernoulli sampling at p=value | 2,396 | **90.4%** |
| latency | Time-to-first-spike (1/value) | 142 | **88.1%** |
| burst | Burst length proportional to value | 1,055 | **84.7%** |
| rank_order | Neurons fire in value-descending order | 142 | **67.1%** |
| phase | Spike phase within oscillation cycle | 3,011 | **62.1%** |

## Key Findings (Kaggle run 2026-03-28)

1. **Rate-based encodings dominate**: repeat_binary, direct, and rate all
   achieve 90%+ with similar spike counts (~2,400). The SNN processes
   repeated static input well.

2. **Latency encoding is the efficiency winner**: 88.1% with only 142
   spikes/sample (17x fewer than rate). Best accuracy-per-spike ratio.

3. **Phase encoding underperforms**: 62.1% despite 3,011 spikes. The
   fixed oscillation pattern doesn't carry enough discriminative information
   for a feedforward SNN.

4. **Rank-order is temporally sparse**: 142 spikes (same as latency) but
   67.1% accuracy. The ordering information alone isn't sufficient without
   temporal processing (recurrent networks would help).

## Pareto-Optimal Encodings

For MNIST classification:
- **Best accuracy**: repeat_binary (91.9%)
- **Best efficiency**: latency (88.1% at 142 spikes)
- **Best balance**: rate (90.4%, standard choice, well-understood)

## Training Setup

- Model: SpikingNet(784 -> 128 -> 10, 1 layer, beta=0.9)
- 5,000 train / 1,000 test samples
- 5 epochs, Adam lr=1e-3, T=25 timesteps
- Total benchmark time: 134s on Kaggle CPU

## Files

- `benchmarks/results/encoding_comparison_results.json` -- full results
- `src/sc_neurocore/encoding/encoders.py` -- all 7 encodings

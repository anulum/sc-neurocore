# NeuroBench SHD Benchmark

First measured accuracy on the Spiking Heidelberg Digits (SHD) task using
SC-NeuroCore's `SpikingNet` with surrogate gradient training.

## Reference

Cramer, B. et al. (2022). The Heidelberg Spiking Data Sets for the Systematic
Evaluation of Spiking Neural Networks. IEEE TNNLS 33(7):2744-2757.

## Dataset

- 20 classes (digits 0-9 in English and German)
- 700 input channels (artificial cochlea)
- ~8,156 train / 2,264 test samples
- Binned to T=100 timesteps (10ms bins over 1s)

## Results (Kaggle run 2026-03-28, CPU)

| Model | Params | Test Acc | Train Acc | Epochs | Inference |
|-------|--------|----------|-----------|--------|-----------|
| SpikingNet(256h, 2L) | 250,388 | **79.28%** | 99.93% | 30 | 1,316 samp/s |
| SpikingNet(128h, 2L) | 108,820 | **77.92%** | 96.73% | 20 | 2,104 samp/s |

## Context

| Method | SHD Accuracy | Source |
|--------|-------------|--------|
| SC-NeuroCore SpikingNet (this) | 79.28% | Measured |
| snnTorch feedforward LIF | ~75-80% | Eshraghian 2023 |
| Heterogeneous recurrent SNN | ~92% | Perez-Nieves 2021 |
| Attention-based SNN | ~95% | Yao 2024 |
| ANN baseline (LSTM) | ~90% | Cramer 2022 |

79% from a feedforward SNN with no recurrence is competitive with snnTorch.
Higher accuracy requires recurrent connections or attention mechanisms.

## NeuroBench Metrics

| Metric | 256h model | 128h model |
|--------|-----------|-----------|
| Parameters | 250,388 | 108,820 |
| Avg spikes/sample | 377 | 343 |
| Synaptic ops/sample | 97,181 | 39,411 |
| Inference latency | 0.76 ms/sample | 0.48 ms/sample |

## Training Details

- Optimizer: Adam, lr=1e-3, cosine annealing
- Loss: CrossEntropyLoss on spike counts
- Surrogate gradient: arctan
- LIF beta: 0.9
- Gradient clipping: max norm 1.0
- Total training time: 926s on Kaggle CPU

## Caveats

- This is a feedforward model. Recurrent architectures achieve 90%+.
- No data augmentation, no regularisation beyond grad clipping.
- T=100 bins (10ms) may lose fine temporal structure.
- CPU training only (Kaggle P100 incompatible with PyTorch 2.10).

## Files

- `benchmarks/results/neurobench_shd_results.json` -- full training history
- `notebooks/neurobench_shd_kaggle.py` -- Kaggle training script

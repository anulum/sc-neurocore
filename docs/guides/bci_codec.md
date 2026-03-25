# BCI Spike Codec

Spike train compression for brain-computer interface telemetry.

## The Problem

High-density neural implants (Neuralink N1: 1,024 electrodes at 20 kHz, 10-bit)
generate ~200 Mbps raw data. Wireless uplinks deliver 10-20 Mbps. The gap
requires 10-200x compression running on-implant under severe constraints:
<10 mW power, <1 ms latency, deterministic worst-case.

Scaling to 3,000-10,000+ channels (Neuralink 2026+ roadmap) makes this 3-10x worse.

## SC-NeuroCore Solution

Two-layer codec architecture:

1. **ISI Codec** (`SpikeCodec`) — baseline compression via inter-spike interval
   encoding with LEB128 variable-length integers. Exploits sparsity: cortical
   neurons fire at 0.5-5 Hz, so >99.9% of time bins are zeros. Achieves 50-200x
   on typical data.

2. **Predictive Codec** (`PredictiveSpikeCodec`) — surprise-only transmission.
   Maintains a per-channel firing rate predictor (exponential moving average).
   XORs actual spikes against predictions. Compresses only the prediction errors.
   Removes structured correlations (bursts, oscillations, drift) that ISI alone
   cannot exploit.

### Architecture Diagram

```
Encoder (on-implant)                    Decoder (external)
┌─────────────────────┐                ┌─────────────────────┐
│ actual spikes (T,N) │                │ compressed bytes    │
│         │           │                │         │           │
│    ┌────▼────┐      │                │    ┌────▼────┐      │
│    │Predictor│◄──┐  │                │    │ISI Decode│      │
│    │ (EMA)   │   │  │                │    └────┬────┘      │
│    └────┬────┘   │  │                │         │           │
│         │predict │  │                │    error matrix     │
│    ┌────▼────┐   │  │                │         │           │
│    │  XOR    │   │  │                │    ┌────▼────┐      │
│    └────┬────┘   │  │                │    │  XOR    │      │
│         │error   │  │                │    └────┬────┘      │
│    ┌────▼────┐   │  │                │         │           │
│    │ISI Encode│  │  │                │    ┌────▼────┐      │
│    └────┬────┘   │  │                │    │Predictor│◄──┐  │
│         │        │  │                │    │ (EMA)   │   │  │
│    compressed   update              │    └────┬────┘   │  │
│    bytes         │  │                │         │      update│
│                  └──┘                │    recovered     │  │
│                                      │    spikes       └──┘ │
└─────────────────────┘                └─────────────────────┘
```

Both sides run identical predictors. Encoder updates predictor with actual spikes.
Decoder recovers actual spikes first (XOR error with prediction), then updates
its predictor with the recovered spikes. Deterministic — no state synchronization
needed.

## Quick Start

```python
import numpy as np
from sc_neurocore.spike_codec import PredictiveSpikeCodec, SpikeCodec

# Generate synthetic spike data: 1024 channels, 1 second at 20 kHz
rng = np.random.RandomState(42)
spikes = (rng.random((20000, 1024)) < 0.001).astype(np.int8)  # ~1 Hz firing

# Baseline ISI codec
baseline = SpikeCodec(mode="lossless")
raw_data, raw_result = baseline.compress(spikes)
print(raw_result.summary())

# Predictive codec — should beat baseline on structured data
codec = PredictiveSpikeCodec(alpha=0.005, threshold=0.5)
pred_data, pred_result = codec.compress(spikes)
print(f"Predictive: {pred_result.compression_ratio:.1f}x, "
      f"prediction accuracy: {pred_result.prediction_accuracy:.1%}, "
      f"error sparsity: {pred_result.error_sparsity:.1%}")

# Lossless roundtrip
recovered = codec.decompress(pred_data, 20000, 1024)
assert np.array_equal(recovered, spikes)
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `alpha` | 0.005 | EMA smoothing factor. Higher = faster adaptation. 0.001-0.01 for 20 kHz. |
| `threshold` | 0.5 | Predicted rate above this → predict spike. Tune to firing rate. |
| `base_mode` | 'lossless' | 'lossless' or 'lossy' for underlying ISI codec. |
| `timing_precision` | 1 | Lossy mode: quantize timing to this resolution. |

## Hardware Mapping

The predictive codec maps to on-implant ASIC:

- **EMA update**: one multiply-accumulate per channel per timestep (or fixed-point shift-add)
- **Threshold compare**: one comparator per channel
- **XOR**: one gate per channel per timestep
- **ISI encoder**: counter + LEB128 shift register per channel

Total gate count for 1024 channels: ~50K gates (excluding ISI FIFO).
Power estimate: <1 mW at 7nm (dominated by SRAM for ISI buffers).

The Verilog RTL in `hdl/` provides verified building blocks:
`sc_aer_encoder.v` (priority encoder), `sc_lif_neuron.v` (Q8.8 fixed-point),
`sc_bitstream_encoder.v` (LFSR comparator).

## API Reference

::: sc_neurocore.spike_codec.predictive_codec.PredictiveSpikeCodec
    options:
      show_source: true
      members:
        - compress
        - decompress

::: sc_neurocore.spike_codec.predictive_codec.PredictiveCompressionResult

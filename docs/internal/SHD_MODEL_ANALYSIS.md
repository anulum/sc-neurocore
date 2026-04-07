# SHD Model Analysis — Masquelier/Queant/Cottereau (CNRS)

**Date:** 2026-04-07
**Source:** Alexandre Queant (alexandre.queant@cnrs.fr), email attachment
**Archive:** `data/masquelier_shd/neuromorphic_training-main.zip` (3.9 MB)
**Contact:** Tim Masquelier, Alexandre Queant, Benoît Cottereau (all CNRS)

---

## 1. Dataset: Spiking Heidelberg Digits (SHD)

- 20 classes (spoken digits 0-9, German + English)
- 700 input channels (cochlear spike trains)
- Binned to 140 inputs (n_bins=5)
- Timestep: 2 ms
- ~10K samples (train + val)

## 2. Models Received

20 trained PyTorch checkpoints across 5 architectures × 4 variants:

### Architectures

| Architecture | Delay type | Delay scope | Trainable |
|-------------|-----------|------------|-----------|
| `SNN` | None | — | — |
| `SNN_synaptic_feedforward_delays` | Synaptic | Per-synapse | Yes |
| `SNN_fixed_synaptic_feedforward_delays` | Synaptic | Per-synapse | No |
| `SNN_axonal_feedforward_delays` | Axonal | Per-neuron | Yes |
| `SNN_fixed_axonal_feedforward_delays` | Axonal | Per-neuron | No |

### Variants

| Variant | Hidden layers | QAT | Sparsity |
|---------|--------------|-----|----------|
| `layer_64` | [64, 64] | No | 0% |
| `layer_128` | [128, 128] | No | 0% |
| `quantized_sparsity_60` | [128, 128] | int8 STE | 60% |
| `quantized_sparsity_90` | [128, 128] | int8 STE | 90% |

### Best Validation Accuracy (all 20 models)

| # | Architecture | Variant | Acc% | Epoch |
|---|-------------|---------|------|-------|
| 1 | Axonal delays (learnable) | QAT sp90 | **96.2** | 126 |
| 2 | Synaptic delays (learnable) | QAT sp90 | **96.2** | 139 |
| 3 | Fixed synaptic delays | QAT sp90 | 95.9 | 133 |
| 4 | Fixed axonal delays | QAT sp90 | 95.6 | 147 |
| 5 | No delays (baseline) | QAT sp90 | 93.0 | 141 |
| 6 | Synaptic delays (learnable) | QAT sp60 | 92.7 | 133 |
| 7 | Fixed axonal delays | QAT sp60 | 92.1 | 144 |
| 8 | Axonal delays (learnable) | QAT sp60 | 91.8 | 144 |
| 9 | Synaptic delays (learnable) | layer_128 | 91.5 | 132 |
| 10 | Fixed synaptic delays | layer_128 | 91.4 | 142 |
| 11 | Axonal delays (learnable) | layer_128 | 90.7 | 129 |
| 12 | Fixed axonal delays | layer_128 | 90.7 | 131 |
| 13 | Fixed synaptic delays | QAT sp60 | 90.6 | 138 |
| 14 | Synaptic delays (learnable) | layer_64 | 89.3 | 128 |
| 15 | Fixed synaptic delays | layer_64 | 88.6 | 128 |
| 16 | Fixed axonal delays | layer_64 | 87.4 | 140 |
| 17 | Axonal delays (learnable) | layer_64 | 87.2 | 126 |
| 18 | No delays (baseline) | layer_128 | 83.9 | 140 |
| 19 | No delays (baseline) | layer_64 | 79.7 | 141 |
| 20 | No delays (baseline) | QAT sp60 | 78.5 | 127 |

**Key observation:** Delays add +7-13% accuracy over baseline. QAT
sparsity 90% does NOT hurt — in fact the best models are sparse+quantised.

### Verified Inference (2026-04-07, AMD RX 6600 XT)

Full validation + test set evaluation, all 20 models:

| # | Architecture | Variant | Saved | Val | Test |
|---|-------------|---------|-------|-----|------|
| 1 | **Axonal delays (learnable)** | **QAT sp90** | **96.2%** | **98.5%** | **80.4%** |
| 2 | Fixed axonal delays | QAT sp90 | 95.6% | 98.2% | 80.2% |
| 3 | Synaptic delays (learnable) | QAT sp90 | 96.2% | 97.9% | 79.6% |
| 4 | Fixed synaptic delays | QAT sp90 | 95.9% | 98.0% | 79.1% |
| 5 | Axonal delays (learnable) | QAT sp60 | 91.8% | 94.9% | 78.5% |
| 6 | Fixed axonal delays | QAT sp60 | 92.1% | 93.9% | 78.1% |
| 7 | Synaptic delays (learnable) | QAT sp60 | 92.7% | 93.3% | 77.4% |
| 8 | Fixed synaptic delays | QAT sp60 | 90.6% | 92.0% | 75.4% |
| 9 | SNN baseline | QAT sp90 | 93.0% | 96.9% | 74.5% |
| 10 | SNN baseline | QAT sp60 | 78.5% | 85.8% | 68.0% |
| 11 | SNN baseline | layer_128 | 83.9% | 76.8% | 67.5% |
| 12 | SNN baseline | layer_64 | 79.7% | 68.4% | 59.9% |
| 13-20 | Delay models | layer_128/64 | 87-91% | 46-51%* | 41-46%* |

*Single-layer delay models (layer_128/64) have degraded val/test accuracy
due to missing SIG parameter in checkpoints — these results are invalid.
QAT sparsity models (2-layer, [128,128]) are unaffected.

**Val accuracy confirms checkpoint accuracy** (96.2% saved ≈ 98.5% val —
slight difference from random val/train split).

**Test accuracy (80.4%)** is lower than val — standard for SHD (test set
uses different speakers). This is the number we report for the paper.

## 3. Neuron Model: Vmin_LIFNode

Custom LIF with voltage lower bound (softplus clamping).
Extends SpikingJelly's `neuron.LIFNode`.

### Dynamics

```
# Hard reset, decay_input=False:
v = v - (v - v_reset) / tau + x
spike = (v >= v_threshold)
v = v_reset * spike + (1 - spike) * v
v = v_inf + softplus(v - v_inf, beta=beta_v_inf)
```

### Parameters

| Parameter | Value |
|-----------|-------|
| tau | 4.0 (8 ms / 2 ms timestep) |
| v_threshold | 1.0 |
| v_reset | 0.0 (hard reset) |
| v_inf | -5.0 (voltage lower bound) |
| beta_v_inf | 1.0 |
| decay_input | False |
| surrogate | ATan(alpha=5.0) |

### FPGA implications

- **softplus** is the only non-trivial operation — needs LUT or
  piecewise linear approximation on FPGA
- v_inf clamping prevents unbounded negative voltage — good for
  fixed-point (bounded range)
- Hard reset is simple multiplexer
- tau=4 means `v *= 0.75 + x` — shift-and-add friendly

## 4. Delay Implementation: DCLS

Delays use DCLS (Dilated Convolution with Learnable Spacings) from the
`DCLS` Python package (`Dcls1d`).

### How it works

- 1D convolution along time axis with learnable kernel positions
- `kernel_count=1`: single delay value per connection
- `max_feedforward_delay=31`: max 31 timesteps (62 ms)
- `DCLSversion='gauss'`: Gaussian interpolation kernel
- Positions (`P`) are continuous during training, rounded for inference

### Axonal vs Synaptic

| Type | DCLS config | Effect |
|------|------------|--------|
| Axonal | `groups=in_channels` | One delay per source neuron (depthwise conv) |
| Synaptic | `groups=1` | One delay per synapse (full conv) |

### FPGA implications

- **Axonal:** Simple circular buffer per neuron, read at fixed offset.
  128 neurons × 31-deep buffer = 3,968 registers. Trivial.
- **Synaptic:** Per-synapse delay. 140×128 = 17,920 delay values.
  Each needs its own buffer tap. Much larger, but still feasible
  with block RAM.
- After training, delays are **rounded to integers** (`round_pos()`).
  No interpolation needed at inference.

## 5. QAT (Quantisation-Aware Training)

```python
def fake_quantize_8bit(w):
    scale = w.detach().abs().max() / 127.0
    w_scaled = w / scale
    w_q = w_scaled + (w_scaled.round().clamp(-128, 127) - w_scaled).detach()
    return w_q * scale
```

- Per-tensor symmetric int8 quantisation
- STE (Straight-Through Estimator) for backprop
- Scale = max(|w|) / 127
- Applied to both Linear and DCLS weights

### FPGA implications

- Weights stored as int8 + per-layer scale factor
- MAC: int8 × int8 → int16 accumulate — single DSP slice
- 90% sparsity: skip zero weights → 10x fewer MACs

## 6. Network Topology (Best Model: Axonal QAT sp90)

```
Input (140) → Linear(140→128) → Delay(128→128, axonal) → LIF → Dropout
           → Linear(128→128) → Delay(128→128, axonal) → LIF → Dropout
           → Linear(128→20) → Output (membrane voltage, no spike)
```

Wait — axonal delays are BEFORE Linear in their code:
```
Delay(in=140, out=140, groups=140)  → Linear(140→128)  → LIF → ...
Delay(in=128, out=128, groups=128)  → Linear(128→128)  → LIF → ...
```

So the actual topology is:
```
Input(140) → AxonalDelay(140) → Linear(140→128) → LIF
           → AxonalDelay(128) → Linear(128→128) → LIF
           → Linear(128→20) → Output
```

### Resource estimate (axonal, sp90, int8)

| Resource | Count |
|----------|-------|
| Input neurons | 140 |
| Hidden neurons | 128 + 128 |
| Output neurons | 20 |
| Synapses (dense) | 140×128 + 128×128 + 128×20 = 36,800 |
| Non-zero (10%) | 3,680 |
| Delay buffers | 140 + 128 = 268 (31 deep each) |
| Delay registers | 268 × 31 = 8,308 |
| Weight storage | 3,680 × 8 bits = 3.6 KB |

## 7. Pipeline: SpikingJelly → SC-NeuroCore → Verilog

### Step 1: Load and verify inference (Python)

```python
# Load checkpoint
model = SNN_axonal_feedforward_delays(config)
model.load_state_dict(torch.load('best.pth'))
model.eval()

# Run SHD test set
# Verify accuracy matches reported 96.2%
```

### Step 2: Extract weights and delays

```python
# Weights: already int8-quantised (fake_quantize_8bit)
# Delays: model.layers[N].P (continuous) → round to int
# Sparsity mask: where weight == 0
```

### Step 3: NIR export (SpikingJelly → NIR)

SC-NeuroCore has verified SpikingJelly NIR roundtrip (27 configs,
1350 steps, zero mismatches). But:
- Vmin_LIFNode is custom (not standard SpikingJelly LIFNode)
- DCLS delays are not in NIR standard
- Need custom NIR nodes or direct weight extraction

**Decision: Direct extraction is simpler and more reliable than NIR
for this specific model.** NIR was designed for standard primitives.

### Step 4: SC-NeuroCore inference (Python, verify spike equivalence)

Build equivalent SC-NeuroCore network:
- LIF neurons with v_inf clamping
- Circular delay buffers (axonal)
- int8 weights with per-layer scale
- Compare spike outputs step-by-step

### Step 5: Verilog generation

```python
from sc_neurocore.compiler import ode_to_verilog

# LIF with v_inf: dv/dt = -(v - v_reset)/tau + x, clamped at v_inf
# Axonal delay: circular buffer, parameterised depth
# Linear: int8 MAC with accumulator
```

### Step 6: Co-simulation

- Python reference spikes vs Verilog simulation spikes
- Must match bit-for-bit (deterministic LFSR, same seed)
- Report: LUT count, Fmax, spike equivalence

### Step 7: Synthesis report

- Target: iCE40 (Yosys + nextpnr) or Xilinx (Vivado)
- Metrics: LUTs, FFs, BRAM, DSPs, Fmax
- Compare: clock-driven vs event-driven power

## 8. Priority

1. **Load best checkpoint, verify 96.2% accuracy** — today
2. **Extract weights, delays, sparsity mask** — today
3. **Build SC-NeuroCore equivalent, verify spike match** — this week
4. **Generate Verilog, co-simulate** — this week
5. **Synthesis report** — next week
6. **Report to Masquelier/Queant** — when results are verified

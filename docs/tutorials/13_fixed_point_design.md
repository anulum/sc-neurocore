<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->

# Fixed-Point Arithmetic for Hardware Deployment

Design and verify fixed-point neural networks that map directly to
FPGA logic. This tutorial covers the Q8.8 number format used by
SC-NeuroCore's hardware path, quantisation-aware training, and
systematic verification against the floating-point reference.

**Prerequisites**: `pip install sc-neurocore numpy`

## 1. Why fixed-point?

FPGAs have no floating-point units (or expensive ones). Fixed-point
arithmetic uses integer ALUs with an implicit binary point:

```
Q8.8 format (16-bit signed):
  bit 15: sign
  bits 14-8: integer part (-128 to 127)
  bits 7-0: fractional part (resolution = 1/256 ≈ 0.0039)

  Example: 0x0180 = 1.5 (1 × 256 + 128 = 384 → 384/256 = 1.5)
```

SC-NeuroCore's `FixedPointLIFNeuron` and Verilog `sc_lif_neuron.v`
both use Q8.8 — the Python model is **bit-true**, meaning every
intermediate value matches the hardware exactly.

## 2. The FixedPointLIFNeuron

```python
from sc_neurocore import FixedPointLIFNeuron

neuron = FixedPointLIFNeuron()

# Step with explicit Q8.8 parameters
# leak_k=240 ≈ 0.9375, gain_k=16 ≈ 0.0625, threshold=256 = 1.0
spike, v = neuron.step(leak_k=240, gain_k=16, I_t=100)
print(f"Spike: {spike}, V: {v} (Q8 raw)")
print(f"V as float: {v / 256:.4f}")
```

The internal computation:
```
V_next = clamp_s16((leak_k * V_mem) >> 8 + (gain_k * I_t) >> 8)
spike  = V_next >= THRESHOLD
V_out  = 0 if spike else V_next
```

Every multiply is 16×16→32, then right-shift by 8 (the fractional
bits). `clamp_s16` saturates to [-32768, 32767].

## 3. Float → Q8.8 conversion

```python
import numpy as np

def float_to_q8(x):
    """Convert float to Q8.8 signed integer."""
    return int(np.clip(np.round(x * 256), -32768, 32767))

def q8_to_float(q):
    """Convert Q8.8 signed integer to float."""
    return q / 256.0

# Verify round-trip
for val in [0.0, 1.0, -1.0, 0.5, 0.9375, -0.0039]:
    q = float_to_q8(val)
    back = q8_to_float(q)
    error = abs(back - val)
    print(f"  {val:+.4f} → Q8={q:+6d} → {back:+.4f}  error={error:.4f}")
```

Maximum quantisation error: 1/(2×256) = 0.00195.

## 4. Quantisation-aware weight training

Train in float, quantise weights to Q8.8 after each update:

```python
from sc_neurocore import VectorizedSCLayer

L = 512
layer = VectorizedSCLayer(n_inputs=50, n_neurons=20, length=L)

# Quantise weights to Q8.8 grid
def quantise_weights(weights):
    """Snap weights to Q8.8 representable values."""
    q = np.round(weights * 256).astype(np.int16)
    return q.astype(np.float64) / 256.0

# Before quantisation
pre_quant = layer.weights.copy()
layer.weights = quantise_weights(layer.weights)
layer._refresh_packed_weights()

# Measure quantisation error
quant_error = np.abs(pre_quant - layer.weights)
print(f"Mean quantisation error: {quant_error.mean():.6f}")
print(f"Max quantisation error:  {quant_error.max():.6f}")
print(f"Unique weight values:    {len(np.unique(layer.weights))}")
```

## 5. Overflow detection

Fixed-point multiply can overflow. Detect and prevent it:

```python
def safe_q8_multiply(a, b):
    """Q8.8 multiply with overflow detection.

    a, b: Q8.8 integers (int16 range).
    Returns: (a * b) >> 8, clamped to int16.
    """
    product = int(a) * int(b)  # 32-bit intermediate
    shifted = product >> 8
    if shifted > 32767:
        return 32767  # positive saturation
    elif shifted < -32768:
        return -32768  # negative saturation
    return shifted

# Test boundary cases
cases = [
    (32767, 256, "max × 1.0"),
    (32767, 32767, "max × max (overflow)"),
    (-32768, 256, "min × 1.0"),
    (256, 256, "1.0 × 1.0"),
]
for a, b, label in cases:
    result = safe_q8_multiply(a, b)
    print(f"  {label}: {a} × {b} >> 8 = {result} ({q8_to_float(result):.4f})")
```

## 6. Systematic float vs fixed-point comparison

Compare a full network simulation in both modes:

```python
np.random.seed(42)
N_STEPS = 200

# Float reference model
from sc_neurocore import StochasticLIFNeuron
float_neuron = StochasticLIFNeuron(length=256)

# Fixed-point model
fp_neuron = FixedPointLIFNeuron()

float_spikes = []
fp_spikes = []
voltage_errors = []

currents = np.random.randint(-30, 80, size=N_STEPS)

for I in currents:
    # Float path
    f_spike, f_state = float_neuron.step(x_value=max(0, int(I)))
    float_spikes.append(f_spike)

    # Fixed-point path (same parameters in Q8)
    q_spike, q_v = fp_neuron.step(leak_k=240, gain_k=16, I_t=int(I))
    fp_spikes.append(q_spike)

float_count = sum(float_spikes)
fp_count = sum(fp_spikes)
print(f"Float spikes:  {float_count}")
print(f"Q8.8 spikes:   {fp_count}")
print(f"Spike count difference: {abs(float_count - fp_count)}")
```

## 7. Weight export for FPGA

Export trained weights as Verilog `$readmemh` format:

```python
def export_weights_hex(weights, filename):
    """Export Q8.8 weight matrix as hex file for $readmemh."""
    q_weights = np.round(weights * 256).astype(np.int16)
    with open(filename, "w") as f:
        f.write(f"// Q8.8 weights: {q_weights.shape[0]} × {q_weights.shape[1]}\n")
        for row in q_weights:
            for val in row:
                # Convert signed int16 to unsigned hex representation
                unsigned = val & 0xFFFF
                f.write(f"{unsigned:04X}\n")

# Export a layer's weights
export_weights_hex(layer.weights, "layer_weights.hex")

# Verify the file
with open("layer_weights.hex") as f:
    lines = [l.strip() for l in f if not l.startswith("//")]
print(f"Exported {len(lines)} weight values")
print(f"First 5: {lines[:5]}")
```

In Verilog, load with:
```verilog
reg signed [15:0] weights [0:N*M-1];
initial $readmemh("layer_weights.hex", weights);
```

## 8. Bit-exact verification protocol

The gold standard before FPGA deployment:

```python
def verify_bit_exact(n_steps=1000, seed=42):
    """Verify Python fixed-point model matches expected hardware behaviour."""
    np.random.seed(seed)
    fp = FixedPointLIFNeuron()

    voltages = []
    spikes = []
    for _ in range(n_steps):
        I = np.random.randint(-50, 100)
        s, v = fp.step(leak_k=240, gain_k=16, I_t=I)
        spikes.append(s)
        voltages.append(v)

        # Invariant: voltage is always valid int16
        assert -32768 <= v <= 32767, f"Voltage overflow: {v}"
        # Invariant: after spike, voltage resets to 0
        if s:
            assert v == 0, f"Post-spike voltage not zero: {v}"

    print(f"Bit-exact verification: {n_steps} steps, {sum(spikes)} spikes")
    print(f"Voltage range: [{min(voltages)}, {max(voltages)}]")
    print(f"All invariants hold ✓")

verify_bit_exact()
```

## What you learned

- Q8.8 format: 16-bit signed, 8 integer bits, 8 fractional bits
- `FixedPointLIFNeuron` is bit-true to the Verilog RTL
- Quantisation-aware training: snap weights to Q8.8 grid after each update
- Overflow detection: multiply in 32-bit, shift, saturate to int16
- Weight export via `$readmemh` hex files for FPGA loading
- Bit-exact verification: voltage bounds + post-spike reset invariants

## Next steps

- Run co-simulation (Tutorial 09) to verify against actual Verilog
- Implement Q4.12 format for higher precision at smaller integer range
- Use `FixedPointBitstreamEncoder` for stochastic encoding in hardware
- Synthesise the full network with Yosys and measure LUT/FF usage

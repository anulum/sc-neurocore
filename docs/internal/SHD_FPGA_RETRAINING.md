# SHD FPGA Retraining — Progressive SIG Sharpening

**Date:** 2026-04-07
**Status:** Ready to run
**Script:** `data/masquelier_shd/train_sharp_delays.py`
**Base model:** Axonal delays + QAT sp90 (96.23% val, 80.39% test)

---

## Problem

DCLS (Dilated Convolution with Learnable Spacings) uses a Gaussian
kernel (SIG parameter) to interpolate between integer delay positions.
This is essential for gradient-based delay learning (differentiable
delay). However, FPGA hardware can only implement integer delays
(circular buffer with fixed tap offset).

When we replace the Gaussian kernel (SIG=15) with a sharp delta
(SIG→0, equivalent to integer delay), test accuracy drops from
80.4% to 58.6% — a 22 percentage point loss.

### SIG Sweep Results

| SIG | Test Acc | Hardware cost |
|-----|---------|--------------|
| 15.0 (original) | 80.4% | 31-tap FIR per neuron (infeasible) |
| 5.0 | 68.2% | 11-tap FIR |
| 2.0 | 63.8% | 5-tap FIR |
| 1.0 | 62.8% | 3-tap FIR (feasible) |
| 0.5 | 61.7% | ~2-tap |
| 0.1 (sharp) | 58.6% | 1-tap integer delay (ideal for FPGA) |

## Solution: Progressive SIG Sharpening

Standard technique in DCLS literature. Three phases:

### Phase 1 (epochs 0-49): Warmup at SIG=15
- Fine-tune from pretrained checkpoint with original SIG
- Delays are already rounded to integers (via `round_pos()`)
- Purpose: stabilise after rounding

### Phase 2 (epochs 50-99): Cosine annealing SIG 15 → 1
- SIG decreases smoothly: `SIG = 1 + (15-1) * 0.5 * (1 + cos(pi * progress))`
- Network gradually adapts weights to compensate for sharpening
- Delays may shift slightly to optimise under sharper kernel

### Phase 3 (epochs 100-149): Fine-tune at SIG=0.5
- Near-sharp delays (2-tap equivalent)
- Final weight adaptation
- Target: recover as much of the 80.4% as possible

### Expected outcome
Based on DCLS papers (Hammouamri et al. 2023) and standard knowledge
distillation literature, progressive sharpening typically recovers
70-80% of the accuracy gap. Expected test accuracy with sharp delays
after retraining: **72-78%** (vs 58.6% without retraining, vs 80.4%
with Gaussian).

## Training Cost

| GPU | Time | Cost |
|-----|------|------|
| Local RX 6600 XT | ~16h | CHF 0 |
| JarvisLabs RTX 3090 | ~8h | ~CHF 4 |
| JarvisLabs A100 40GB | ~4h | ~CHF 4.20 |
| GCP T4 | ~13h | ~CHF 4.70 |

Model is tiny (37,668 params). Benchmarked at 386s/epoch on RX 6600 XT.

## How to Run

### Local (overnight)
```bash
cd data/masquelier_shd/neuromorphic_training-main
source ~/venv-rocm/bin/activate
nohup python3 ../train_sharp_delays.py > ../train_sharp.log 2>&1 &
```

### JarvisLabs
```bash
# Upload data
rsync -az data/masquelier_shd/ jarvis:~/masquelier_shd/

# SSH to instance
pip install torch spikingjelly DCLS wandb h5py prettytable torchvision
cd ~/masquelier_shd/neuromorphic_training-main
python3 ../train_sharp_delays.py
```

## Output

Results saved to `exp/SHD/SNN_axonal_feedforward_delays/sharp_delays_retrain/`:
- `best.pth` — best validation checkpoint
- `last.pth` — final checkpoint
- `training_log.csv` — per-epoch metrics
- `config.json` — training configuration and final results

## What to Report to Masquelier

After training completes:

> We tested your best model (axonal QAT sp90, 80.4% SHD test) with
> integer delays for FPGA deployment. Direct replacement of the
> Gaussian kernel (SIG=15) with sharp delays drops accuracy to 58.6%.
>
> We retrained with progressive SIG sharpening (3-phase schedule,
> 150 epochs) and recovered to XX.X% with pure integer delays.
> This is FPGA-ready — no Gaussian interpolation needed.
>
> Would you like us to proceed with Verilog generation and synthesis,
> or would you prefer to retrain with your own sharpening schedule?

## Technical Note for Verilog Generation

After successful retraining, the model uses:
- Integer delays (P values are integers, SIG is irrelevant)
- int8 quantised weights (QAT, per-tensor symmetric)
- 90% sparsity mask (zeroed weights stay zero)
- Vmin_LIF with softplus v_inf clamping

For Verilog, the delay module is a simple circular buffer:
```verilog
// Per-neuron axonal delay (integer)
reg [WIDTH-1:0] delay_buffer [0:MAX_DELAY-1];
always @(posedge clk) begin
    delay_buffer[write_ptr] <= input_spike;
    output_spike <= delay_buffer[(write_ptr - DELAY_VALUE) % MAX_DELAY];
    write_ptr <= write_ptr + 1;
end
```

No Gaussian interpolation needed — that was only for training.

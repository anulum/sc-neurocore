<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->

# Tutorial 40: One-Command FPGA Deployment

SC-NeuroCore deploys SNN models to FPGA in one command. Input: a trained
model (NIR or PyTorch). Output: a complete synthesis project with Verilog,
build scripts, and HDL library — ready for Yosys or Vivado.

## Quick Start

```bash
# Deploy a NIR model to Lattice iCE40
sc-neurocore deploy model.nir --target ice40 -o build/

# Deploy to Xilinx Artix-7
sc-neurocore deploy model.nir --target artix7 -o build/

# Deploy a PyTorch state_dict
sc-neurocore deploy weights.pt --target zynq -o build/
```

## What Gets Generated

```
build/
  sc_deploy_lif.sv       Generated neuron module (Q8.8 fixed-point)
  hdl/                   SC-NeuroCore Verilog library (19 modules)
    sc_lif_neuron.v      Q8.8 LIF core
    sc_bitstream_encoder.v  LFSR encoder
    sc_dense_layer_core.v   Dense layer pipeline
    sc_aer_encoder.v     Event-driven AER encoder
    sc_event_neuron.v    Event-triggered LIF
    ...
  Makefile               Yosys build script (ice40/ecp5)
  project.tcl            Vivado build script (artix7/zynq)
  README.md              Build instructions
```

## Supported Targets

| Target | FPGA | Tool | Build command |
|--------|------|------|---------------|
| `ice40` | Lattice iCE40 HX8K | Yosys + nextpnr | `make synth` |
| `ecp5` | Lattice ECP5-85K | Yosys + nextpnr | `make synth` |
| `artix7` | Xilinx Artix-7 100T | Vivado | `vivado -mode batch -source project.tcl` |
| `zynq` | Xilinx Zynq 7020 | Vivado | `vivado -mode batch -source project.tcl` |

## Pipeline Stages

```
[1/5] Load model (NIR graph or PyTorch state_dict)
[2/5] Quantize weights to Q8.8 fixed-point
[3/5] Generate SystemVerilog neuron module
[4/5] Copy 19 HDL library modules
[5/5] Generate target-specific project files
```

## From NIR

Any model exported to NIR (from Norse, snnTorch, SpikingJelly, etc.)
can be deployed:

```python
# Export from SpikingJelly
from spikingjelly.activation_based.nir_exchange import export_to_nir
graph = export_to_nir(model, torch.randn(1, n_input), dt=1e-4)
nir.write("model.nir", graph)
```

```bash
# Deploy to FPGA
sc-neurocore deploy model.nir --target artix7 --dt 1e-4 -o build/
```

## From PyTorch

Save the model's state_dict (not the full model):

```python
torch.save(model.state_dict(), "weights.pt")
```

```bash
sc-neurocore deploy weights.pt --target ice40 --T 256 -o build/
```

The deploy command reconstructs the model architecture from weight shapes
and converts to an SNN using the conversion engine.

## Bitstream Length

The `--T` flag sets the SC bitstream length (default 256):

```bash
sc-neurocore deploy model.nir --target ice40 --T 512 -o build/
```

Longer bitstreams give higher precision at the cost of more clock cycles.
See Tutorial 19 for the precision-latency tradeoff.

## Further Reading

- [Tutorial 38: ANN-to-SNN Conversion](38_ann_to_snn_conversion.md) — conversion pipeline
- [Tutorial 09: Hardware Co-simulation](09_hardware_cosimulation.md) — verify Python vs Verilog
- [Hardware Guide](../hardware/HARDWARE_GUIDE.md) — FPGA deployment details

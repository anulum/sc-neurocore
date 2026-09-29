<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->

# Tutorial 40: One-Command FPGA Deployment

`sc-neurocore deploy` scaffolds an FPGA synthesis project in one command: a
generic LIF neuron module, the SC-NeuroCore HDL library and build scripts for
Yosys or Vivado. The generated RTL is a template; it does not carry a model's
trained weights. For a PyTorch checkpoint the command also converts the trained
dense network and exports exactly that network, and with calibration samples it
measures how the network fits the target's fixed-point format.

## Quick Start

```bash
# Deploy a NIR model to Lattice iCE40
sc-neurocore deploy model.nir --target ice40 -o build/

# Deploy to Xilinx Artix-7
sc-neurocore deploy model.nir --target artix7 -o build/

# Deploy a PyTorch state_dict (its SHA-256 is required)
sc-neurocore deploy weights.pt --checkpoint-sha256 "$(sha256sum weights.pt | cut -d' ' -f1)" \
  --target zynq -o build/
```

## What Gets Generated

```
build/
  sc_deploy_lif.sv       Generic LIF neuron template (Q8.8), not the model's weights
  converted_network.npz  PyTorch input only: the converted dense IF network
  converted_network.json Its manifest: source digest, layers, T, network digest
  target_report.json     With --calibration: fixed-point fit for the target
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
[1/5] Load model (NIR graph, or convert a trusted PyTorch checkpoint)
[2/5] Calibrate the converted network for the target format (with --calibration)
[3/5] Generate the generic LIF RTL template
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
numpy.save("calibration.npy", validation_inputs)  # optional, values in [0, 1]
```

```bash
sc-neurocore deploy weights.pt \
  --checkpoint-sha256 "$(sha256sum weights.pt | cut -d' ' -f1)" \
  --calibration calibration.npy --target ice40 --T 256 -o build/
```

A plain `state_dict` must be a dense ReLU chain: its layers are rebuilt in the
order they were registered, with their trained biases, and any other parameter
(a convolution, a normalisation, a QCFS threshold) is refused instead of being
dropped. The ReLU thresholds come from the calibration samples, or unit scales
without them. A Studio `qcfs_conversion` checkpoint (`training/model_state.pt`)
is rebuilt from its recorded configuration and converted with its learned
thresholds and its own timestep budget; its exported network has the same
digest as the run's `training/conversion_report.json`. A Studio spiking
checkpoint is refused, since it is already a spiking network.

`converted_network.npz` reloads with
`sc_neurocore.conversion.converted_io.load_converted_network`, which refuses a
file whose contents do not match the recorded digest.

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

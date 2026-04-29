<!--
SPDX-License-Identifier: AGPL-3.0-or-later
Commercial license available
© Concepts 1996–2026 Miroslav Šotek. All rights reserved.
© Code 2020–2026 Miroslav Šotek. All rights reserved.
ORCID: 0009-0009-3560-0851
Contact: www.anulum.li | protoscience@anulum.li
SC-NeuroCore — Five-minute FPGA deployment cookbook
-->

# FPGA Deploy Cookbook

This cookbook gives a short, toolchain-light path from a trained model artefact
to a generated FPGA project. It deliberately separates three stages:

1. create or obtain a model artefact;
2. scaffold the FPGA project;
3. optionally run synthesis and parse real reports.

You can complete stages 1-2 without Vivado, Quartus, Yosys, or an FPGA board.

## 1. Install the minimum environment

```bash
python -m venv .venv
. .venv/bin/activate
pip install sc-neurocore
```

Add extras only when the model source needs them:

```bash
pip install "sc-neurocore[nir]"       # NIR import/export
pip install "sc-neurocore[training]"  # PyTorch state_dict path
pip install "sc-neurocore[bioware]"   # MNIST source example dependencies
```

## 2. Create a model artefact

For MNIST-style experiments, use the repository example from a source checkout:

```bash
python examples/mnist_fpga/demo.py --export-verilog build/mnist_weights.vh
```

That path trains on the scikit-learn digits dataset, quantises weights to Q8.8,
and exports Verilog constants. It is useful for inspecting the generated RTL
without requiring an external synthesis tool.

If you already have a NIR graph, skip the example and use the graph directly:

```bash
sc-neurocore deploy model.nir --target ice40 -o build/fpga_scaffold
```

If you have a PyTorch `state_dict`, install the training extra and scaffold from
the weight file:

```bash
sc-neurocore deploy weights.pt --target ice40 --T 256 -o build/fpga_scaffold
```

## 2a. MNIST to NIR to FPGA handoff

For a source checkout, the bundled MNIST training helper saves a PyTorch
checkpoint plus metadata:

```bash
pip install "sc-neurocore[training]" torchvision
python tools/train_pretrained_mnist.py \
  --epochs 1 \
  --output build/mnist/conv_spiking_net_mnist.pt
```

That checkpoint can be scaffolded directly, without an FPGA toolchain:

```bash
sc-neurocore deploy build/mnist/conv_spiking_net_mnist.pt \
  --target ice40 \
  --T 256 \
  -o build/mnist_fpga_scaffold
```

If the MNIST model is trained in a NIR-native frontend such as SpikingJelly,
snnTorch, or Norse, export the trained model to `build/mnist/mnist.nir` first,
then use the same deploy command on the NIR file:

```python
import nir
import torch
from spikingjelly.activation_based.nir_exchange import export_to_nir

example_input = torch.randn(1, 1, 28, 28)
graph = export_to_nir(model, example_input, dt=1e-4)
nir.write("build/mnist/mnist.nir", graph)
```

```bash
sc-neurocore deploy build/mnist/mnist.nir \
  --target ice40 \
  --dt 1e-4 \
  --T 256 \
  -o build/mnist_fpga_scaffold
```

Use the NIR route when the training frontend owns the model definition. Use the
checkpoint route when the SC-NeuroCore training module owns the model
definition. Both routes stop at a generated hardware project until a real
synthesis tool is installed.

## 3. Inspect the scaffold

The deploy command writes a self-contained project directory:

```text
build/fpga_scaffold/
  sc_deploy_lif.sv
  hdl/
  Makefile        # ice40/ecp5 targets
  README.md
```

For Xilinx targets it writes `project.tcl` instead of a Makefile:

```bash
sc-neurocore deploy model.nir --target artix7 -o build/artix7_scaffold
```

At this point the handoff artefact exists. No external FPGA tool has been
invoked unless it is already installed and the target flow supports automatic
open-source synthesis.

## 4. Run synthesis only when a toolchain is installed

Open-source Lattice flow:

```bash
cd build/fpga_scaffold
make synth
```

Xilinx flow:

```bash
cd build/artix7_scaffold
vivado -mode batch -source project.tcl
```

Do not copy resource or power numbers from examples into reports. Treat only
tool-generated outputs from your machine as evidence.

## 5. Parse real reports into optimiser evidence

After Vivado, Quartus, or Yosys produces reports, convert the measured data into
SC design optimiser evidence. The report collector requires explicit design
metadata and measured accuracy so it cannot invent missing evidence.

Create a compact network manifest for the deployed model:

```json
{
  "layers": [
    {"id": "encoder", "mac_count": 256, "is_critical_path": true},
    {"id": "decoder", "mac_count": 192}
  ]
}
```

```bash
sc-neurocore collect-synthesis \
  --design build/network_design.json \
  --utilisation build/fpga_scaffold/reports/utilisation.rpt \
  --power build/fpga_scaffold/reports/power.rpt \
  --timing build/fpga_scaffold/reports/timing.rpt \
  --accuracy-score 0.991 \
  --clock-mhz 100 \
  --inferences-per-run 1 \
  --out build/synthesis_observations.json
```

For the MNIST scaffold, keep the measured accuracy beside the checkpoint
metadata and pass that value to `--accuracy-score`. If a power report does not
exist yet, stop here; do not substitute README or benchmark-table numbers.

```bash
python tools/optimise_sc_design.py \
  --network build/network_design.json \
  --evidence build/synthesis_observations.json \
  --max-luts 50000 \
  --max-power-mw 500 \
  --out build/sc_design_plan.json
```

The JSON plan records selected bitstream lengths, decorrelators, precision,
estimated resource totals, feasibility, rejected layers, and the number of
training points used by the surrogate. Use it as the handoff into later
training, NAS, or hardware-aware deployment loops.

## Checklist

- Base package installed before optional extras.
- Model artefact exists (`.nir`, `.pt`, or generated Verilog weights).
- FPGA scaffold generated under `build/`.
- Tool-generated synthesis reports kept separate from estimates.
- Optimiser evidence includes measured accuracy and design metadata.

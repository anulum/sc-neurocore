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

After Vivado or Quartus produces reports, feed the measured data into the SC
design optimiser. The report parser requires explicit design metadata and
measured accuracy so it cannot invent missing evidence.

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
python tools/optimise_sc_design.py \
  --network build/network_manifest.json \
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

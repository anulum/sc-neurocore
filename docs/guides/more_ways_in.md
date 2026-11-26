<!--
SPDX-License-Identifier: AGPL-3.0-or-later
Commercial license available
© Concepts 1996–2026 Miroslav Šotek. All rights reserved.
© Code 2020–2026 Miroslav Šotek. All rights reserved.
ORCID: 0009-0009-3560-0851
Contact: www.anulum.li | protoscience@anulum.li
SC-NeuroCore — More ways in
-->

# More Ways In

This text was part of the repository README until the README became a short landing page.

## Zenith Quickstart

Train plasticity rules with PyTorch surrogate autograd, and run the same rules in the Rust
backend, whose layer state saves to and loads from a file. The Rust layer starts from one
scalar weight; no call yet copies trained Torch weights into it.

```python
from sc_neurocore.plasticity import create_plasticity_layer
from sc_neurocore._native.learning_bridge import RULE_STDP

# 1. Train the rule layer with PyTorch autograd (needs the training extra)
torch_layer = create_plasticity_layer(count=128, rule_type=RULE_STDP, backend="torch", autograd=True)
# ... standard cross-entropy loss.backward() loop

# 2. Run the same rule in the Rust backend and save its state to a file
rust_layer = create_plasticity_layer(count=128, rule_type=RULE_STDP, backend="rust", weight=0.5)
rust_layer.step(pre_spikes=[True] * 128, post_spikes=[False] * 128, rewards=[0.0] * 128, dt=0.001)
rust_layer.save("rule_layer.scal")
```
See the full end-to-end integration demo in [`examples/zenith_hybrid_resnet.py`](https://github.com/anulum/sc-neurocore/blob/main/examples/zenith_hybrid_resnet.py).

## Bio-hybrid Closed-Loop with ArcaneZenith (Experimental)

The experimental `bioware` module validates finite MEA frames and maps them through spike detection, 16-bit AER epochs, deterministic stochastic bitstreams, and optogenetic pulse proposals. It includes PCA/K-Means spike sorting, heuristic culture monitoring, an onset-gain pharmacology prototype, and an optional `ArcaneZenithCognitiveCore` callback. It is research software, not a clinical or tissue-safety controller; see [the maintained contract](../api/bioware.md).

```bash
# Add only the extras needed for the current workflow.
pip install "sc-neurocore[minimal]"   # explicit v4 minimal profile
pip install "sc-neurocore[core]"      # explicit base profile
pip install "sc-neurocore[nir]"       # NIR interop
pip install "sc-neurocore[training]"  # PyTorch-backed training
pip install "sc-neurocore[studio]"    # local web studio
pip install "sc-neurocore[full]"      # local research environment only
```

See [Install Profiles](install_profiles.md) for the full optional
dependency matrix and research-only boundaries.

## Visual SNN Design Studio (Experimental)

> **Status:** Development preview. Selected workflows are available for
> evaluation, while production hardening continues. API and UI may change
> between releases until the reviewed stable API freeze.

A web interface for designing, training, compiling, and inspecting hardware
evidence for spiking neural networks. Physical FPGA deployment requires
separate board validation.

```bash
pip install sc-neurocore[studio]
sc-neurocore studio              # opens browser at http://127.0.0.1:8001
```

| Feature | What it does |
|---------|-------------|
| **Model Browser** | Browse every registered catalogue entry by category (count in the capability snapshot of the [README](https://github.com/anulum/sc-neurocore#version-and-capability-snapshot)), simulate with parameter sliders |
| **Analysis Views** | Trace, phase portrait, ISI, f-I curve, bifurcation, heatmap, STA, frequency response, characterisation dashboard |
| **Compiler Inspector** | Lower authored equations and numeric profiles to native IR, inspect graph structure, replay bound source and emit SystemVerilog |
| **Synthesis Dashboard** | One-click Yosys synthesis to ice40/ECP5/Gowin/Xilinx, multi-target comparison, resource bars |
| **Training Monitor** | Live loss/accuracy curves via SSE, 6 surrogate gradients, per-layer spike rates |
| **Network Canvas** | Drag-and-drop populations and projections (React Flow), NIR export/import |
| **Full Pipeline** | Network → simulate → compile → synthesise in one click |
| **Project Save/Load** | Persistent workspaces as JSON, server-side storage |

For measured comparisons and their evidence limits, see
[Cross-Framework Evidence](../benchmarks/cross_framework.md).

User manual: [SNN Studio](../studio/manual/index.md).
Technical overview: [Studio Guide](https://anulum.github.io/sc-neurocore/studio/)

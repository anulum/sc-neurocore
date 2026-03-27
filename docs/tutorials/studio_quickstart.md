# Tutorial: Visual SNN Studio Quickstart

This tutorial walks through the complete Studio workflow: browse neuron
models, design a network, train it, compile to Verilog, and estimate
FPGA resources — all from a browser.

**Time:** 10 minutes
**Prerequisites:** `pip install sc-neurocore[studio]`

## 1. Launch the Studio

```bash
sc-neurocore studio
```

Your browser opens at `http://127.0.0.1:8001`. The interface starts in
**Model mode** with the first of 118 neuron models selected and a
live voltage trace displayed.

## 2. Explore Neuron Models

The left panel lists all models by category. Click any model to load it:

1. Click **AdExNeuron** — an adaptive exponential integrate-and-fire model
2. Adjust the **current** slider (header) to 15.0 nA
3. Watch the trace update live — you should see adaptation: initial
   burst followed by regular tonic spiking
4. Click **Char.** to run a one-click characterisation
5. The Char tab shows: firing pattern, f-I curve, top sensitivities

## 3. Compare Models

1. Click **Multi** tab in the tab bar
2. Select 2-3 models from the multi-model picker (left panel):
   - LIFNeuron (simple, fast)
   - AdExNeuron (adaptive)
   - HodgkinHuxleyNeuron (biophysical)
3. All traces overlay in one plot for direct comparison

## 4. Write a Custom ODE

1. Switch to **ODE** mode (top toggle)
2. Select the **Hodgkin-Huxley** template from the dropdown
3. The Monaco editor shows the four coupled ODEs
4. Adjust parameters via sliders — tau_m, C, E_L
5. Click **Bif** to generate a bifurcation diagram of a selected parameter

## 5. Design a Network

1. Click **Canvas** tab (or the Canvas button)
2. Click **+ Exc** to add an excitatory population (80 neurons)
3. Click **+ Inh** to add an inhibitory population (20 neurons)
4. Drag from the excitatory node handle to the inhibitory node to create
   a projection
5. Drag from inhibitory back to excitatory for reciprocal inhibition
6. Click **Simulate** — the status bar shows spike count and firing rates

## 6. Train with Surrogate Gradients

1. Click **Train** tab
2. Set dataset to **Synthetic** (fast, for demo)
3. Set epochs to **5**, surrogate to **atan_surrogate**
4. Click **Train**
5. Watch loss and accuracy curves update live as each epoch completes
6. Layer spike rate bars show activity per spiking layer
7. Click **Stop** to abort early if needed

## 7. Compile to Verilog

1. Switch to **ODE** mode
2. Select the **LIF** template
3. Click **IR** — the Compiler Inspector shows the SC Intermediate
   Representation with a verification badge
4. Click **SV** — SystemVerilog source appears in the right pane
5. This Verilog is synthesisable — it maps the ODE to Q8.8 fixed-point
   hardware

## 8. Synthesise to FPGA

1. Click **FPGA** tab
2. Select target: **ice40** (iCE40 UP5K)
3. Click **Synthesise** (requires Yosys installed)
4. Resource bars show LUT, FF, BRAM, DSP utilisation
5. Click **All Targets** for a side-by-side comparison across all 4 targets
6. If Yosys is not installed, click **Estimate** for a heuristic from the IR

## 9. Full Pipeline

1. Go back to **Canvas** tab
2. With your network designed, click **Pipeline → ICE40**
3. The pipeline chains: validate → simulate → compile → synthesise
4. Status bar shows each step's result

## 10. Save Your Work

1. In the left sidebar under **Projects**, click **Save**
2. Enter a project name
3. The entire state — equations, parameters, network graph, synthesis
   target — is saved as JSON on the server
4. Click **Refresh** to see saved projects, click a name to restore

## What's Next

- [Studio Guide](../guides/studio.md) — full API reference, all 18+ views, protocols
- [Synthesis Dashboard](../studio/synthesis-dashboard.md) — FPGA targets, multi-target details
- [Training Monitor](../studio/training-monitor.md) — surrogates, cell types, SSE streaming
- [Network Canvas](../studio/network-canvas.md) — NIR format, population/projection API
- [Integration](../studio/integration.md) — pipeline details, project storage format

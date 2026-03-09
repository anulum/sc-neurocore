# SC-NeuroCore — Market Readiness TODO

Status: Draft 2026-03-08

## P0 — Without these, nobody will use it

- [ ] **JOSS paper submission** — 2-page software paper, peer-reviewed, citeable. Follow https://joss.theoj.org/about. Requires: statement of need, references to similar software, mention of ongoing research projects using it.
- [ ] **FPGA deployment proof** — Deploy a 1000-neuron LIF network on Xilinx Artix-7 or Zynq. Measure and publish: LUT count, BRAM usage, DSP slices, Fmax, dynamic power (W), latency per timestep. This is the moat.
- [ ] **Brian2 head-to-head benchmark** — Same Brunel network, same parameters, wall-clock comparison at 1K/10K/100K neurons. Show where SC-NeuroCore wins (SC mode, RTL pipeline, Rust engine).
- [ ] **Trim README to ~50 lines** — Quick start, one code example, one diagram, badge row, link to full docs. Move everything else to docs site.

## P1 — Credibility and adoption multipliers

- [ ] **Kill frontier/speculative tiers in wheel** — Remove generative, world_model, analysis, audio, dashboard, viz, swarm, research/ from `pip install sc-neurocore`. Keep them source-only. Fewer modules = stronger signal.
- [ ] **One killer tutorial** — "Deploy an SNN on FPGA in 20 minutes with SC-NeuroCore." End-to-end: Python model → co-sim → Verilog → synthesis. Publish on docs site + dev.to/Medium.
- [ ] **FPGA synthesis reports in repo** — Vivado/Quartus reports committed as artifacts. LUT/BRAM/DSP/Fmax/power for reference designs.
- [ ] **Contextualise "512x real-time" claim** — State baseline explicitly: "512x vs pure-Python simulation at N=10K, L=1024." Add Brian2 Cython comparison at same scale.
- [ ] **MNIST-on-FPGA demo** — SC-encoded inference of a small classifier on FPGA. Measured accuracy, power, latency. Publishable result.

## P2 — Community and discoverability

- [ ] **Awesome-neuromorphic listing** — PR to awesome-snn / awesome-neuromorphic lists on GitHub.
- [ ] **Conference lightning talk** — NICE (Neuro-Inspired Computational Elements), ICONS, or Telluride workshop. 5-minute demo of Python→Verilog pipeline.
- [ ] **Target lab outreach** — Email 5 neuromorphic hardware labs with a 3-sentence pitch + link. Targets: ETH Zurich (Indiveri), TU Dresden (Mayr), Georgia Tech (Rozell), Intel Neuromorphic, IBM Research.
- [ ] **GitHub Discussions enabled** — Already listed in README. Seed with a "Show & Tell" and "Q&A" category.
- [ ] **Publish wheels for sc_neurocore_engine** — Configure trusted publisher for the Rust engine package. Users currently can't `pip install` the fast path.

## P3 — Nice to have

- [ ] **Nature Electronics letter** — If FPGA proof shows ≥5x energy efficiency vs ANN equivalent, write it up. High impact, right audience.
- [ ] **Loihi/SpiNNaker comparison** — Benchmark against Intel Lava on equivalent network. Different hardware philosophy but same audience.
- [ ] **hls4ml interop** — Bridge or comparison showing SC advantages over HLS-based neural network deployment.
- [ ] **Power analysis paper** — Theoretical + measured analysis of stochastic computing power advantages for neuromorphic edge inference.

## P4 — Post-Audit Benchmark Re-Runs (2026-03-09)

Results from JarvisLabs A6000 re-run (2026-03-09):

- [x] **Re-run Brunel v5_izhikevich** — DONE. 13,230 spikes @ 13.2 Hz (was 15,331 @ 15.3 Hz). 14% decrease from half-step integration fix. Updated in `snn_translator_20v.json`.
- [ ] **Re-run Brunel v12_stdp_lif** — JarvisLabs produced 0 spikes (env artifact — Python STDP was NOT changed in audit). Must re-run on UpCloud EPYC (original benchmark env).
- [ ] **Re-run Rust Criterion kuramoto** — JarvisLabs: 40.86 ms (Xeon 4216), not comparable to UpCloud 118.7 ms (EPYC 9575F). Must re-run on UpCloud.
- [ ] **Spot-check Brunel scaling** — JarvisLabs produced 0 spikes (env artifact — StochasticLIFNeuron untouched). Must re-run on UpCloud.

## Anti-patterns to avoid

- Don't add more modules before getting users for existing ones
- Don't write more benchmarks before publishing one hardware result
- Don't expand scope before the core story is proven
- Don't count test coverage as a substitute for external validation

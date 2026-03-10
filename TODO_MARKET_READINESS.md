# SC-NeuroCore — Market Readiness TODO

Status: v3.10.0 (2026-03-10)

## P0 — Without these, nobody will use it

- [x] **JOSS paper written** — DONE (v3.10.0). `paper/paper.md` + `paper.bib`, 12 refs with DOIs, MNIST results, Brian2 comparison, formal verification. Submission-ready for https://joss.theoj.org/papers/new.
- [ ] **JOSS paper submitted** — Submit via JOSS web form. Requires: public repo (done), statement of need (done), functioning software (done), test suite (done). Estimated review: 4-8 weeks.
- [ ] **FPGA deployment proof** — Deploy a 1000-neuron LIF network on Xilinx Artix-7 or Zynq. Measure and publish: LUT count, BRAM usage, DSP slices, Fmax, dynamic power (W), latency per timestep. This is the moat. Vivado tooling ready (`tools/vivado_impl.tcl`), need physical board.
- [x] **Brian2 head-to-head benchmark** — DONE. UpCloud EPYC 9575F (2026-03-10): V18 Numba **4.0x** faster at 1K, Brian2 **1.35x** faster at 10K. V21 sparse 0.47s at 1K. Rates match (~100 Hz at 1K, ~41 Hz at 10K). Results: `benchmarks/results/upcloud_p4_rerun_20260310/`.
- [x] **Trim README to ~50 lines** — DONE. 359→166 lines. Benchmarks section, 10 HDL modules, MNIST demo, badge row, architecture diagram.

## P1 — Credibility and adoption multipliers

- [x] **Kill frontier/speculative tiers in wheel** — DONE. 14 frontier modules excluded from wheel via `pyproject.toml` `[tool.setuptools.packages.find] exclude`. Wheel contains only core SC+SNN+FPGA story.
- [x] **One killer tutorial** — DONE. `docs/tutorials/fpga_in_20_minutes.md`. End-to-end: train → quantise → SC simulate → synthesise → Vivado. 6 sections with scaling guide.
- [x] **FPGA synthesis reports in repo** — DONE. Yosys CI + sv2v preprocessing. `sc_neurocore_top` (3x7) = 7,382 LUTs. Vivado TCL script + report parser added (`tools/vivado_impl.tcl`, `tools/vivado_report.py`). Fmax/power require Vivado hardware.
- [x] **Contextualise "512x real-time" claim** — DONE. README now has Benchmarks section with Brian2 comparison table (5.2s Numba vs 1.6s Brian2 at 1K), Rust SIMD throughput, Yosys synthesis results. Honest framing: SC targets FPGA-scale networks, Brian2 faster at N>1K.
- [x] **MNIST-on-FPGA demo** — DONE. `examples/mnist_fpga/demo.py`. Float 94.2%, Q8.8 94.2%, SC 94.0% (L=1024). 16→10 config = ~56K LUTs (Artix-7 100T). Verilog weight export + `sc_dense_matrix_layer.v` HDL module.

## P2 — Community and discoverability

- [ ] **Awesome-neuromorphic listing** — PR to awesome-snn / awesome-neuromorphic lists on GitHub.
- [ ] **Conference lightning talk** — NICE (Neuro-Inspired Computational Elements), ICONS, or Telluride workshop. 5-minute demo of Python->Verilog pipeline.
- [ ] **Target lab outreach** — Email 5 neuromorphic hardware labs with a 3-sentence pitch + link. Targets: ETH Zurich (Indiveri), TU Dresden (Mayr), Georgia Tech (Rozell), Intel Neuromorphic, IBM Research.
- [ ] **GitHub Discussions enabled** — Already listed in README. Seed with a "Show & Tell" and "Q&A" category.
- [ ] **Publish wheels for sc_neurocore_engine** — Configure trusted publisher for the Rust engine package. Users currently can't `pip install` the fast path.

## P1.5 — Competitive Gap Closures

- [x] **GPU SNN training with surrogate gradients** — DONE. `sc_neurocore.training` module: 3 surrogate gradient functions (FastSigmoid, SuperSpike, ATan), PyTorch `nn.Module` SNN layers (LIFCell, RecurrentLIFCell, SpikingNet), training loops, 3 loss functions. MNIST example: ~95% accuracy in 10 epochs. `to_sc_weights()` bridges float training to SC bitstream deployment. Closes the Norse/snnTorch competitive gap. 31 tests. `pip install sc-neurocore[training]`.

## P3 — Nice to have

- [ ] **Nature Electronics letter** — If FPGA proof shows >=5x energy efficiency vs ANN equivalent, write it up. High impact, right audience.
- [ ] **Loihi/SpiNNaker comparison** — Benchmark against Intel Lava on equivalent network. Different hardware philosophy but same audience.
- [ ] **hls4ml interop** — Bridge or comparison showing SC advantages over HLS-based neural network deployment.
- [ ] **Power analysis paper** — Theoretical + measured analysis of stochastic computing power advantages for neuromorphic edge inference.
- [x] **Sparse weight matrix (scipy.sparse)** — DONE (v3.10.0). VectorizedSCLayer `sparse=True` uses CSR backend. V21 sparse Numba Brunel: 3x faster than Brian2 at 1K, memory 10x lower at 10K. Brian2 still wins at 10K speed (compiled C++ codegen).

## P4 — Post-Audit Benchmark Re-Runs (COMPLETE 2026-03-10)

All re-runs completed on UpCloud EPYC 9575F (HICPU-8xCPU-16GB, fi-hel2).
Results: `benchmarks/results/upcloud_p4_rerun_20260310/`

- [x] **Re-run Brunel v5_izhikevich** — DONE. 13,230 spikes @ 13.2 Hz (was 15,331 @ 15.3 Hz). 14% decrease from half-step integration fix. Updated in `snn_translator_20v.json`.
- [x] **Re-run Brunel v12_stdp_lif** — DONE (UpCloud). v12: 186s, 99.9 Hz. V18: 1.11s, 100.0 Hz. V21 sparse: 0.47s, 100.0 Hz. All rate-matched to Brian2 98.7 Hz.
- [x] **Re-run Rust Criterion kuramoto** — DONE (UpCloud). 65.9 ms (was 118.7 ms). 1.8x faster with 1/N normalization fix.
- [x] **Spot-check Brunel scaling** — DONE (UpCloud). SC dense 11.9x at 1K, 2.8x at 5K vs Brian2. Rates match.

## P5 — Brian2 Benchmark: Correlated-Drive Discovery (2026-03-09)

Initial Brian2 benchmark showed a 3x firing rate divergence at 10K neurons (Brian2: 119 Hz, SC: 41 Hz) and apparent 14.5x SC speedup at 1K.

**Root cause**: Brian2's `PoissonGroup` + `Synapses.connect()` (all-to-all) creates shared external spike sources that fire into ALL neurons simultaneously. This introduces artificial positive correlations in external drive, inflating firing rates at large N via synchronized threshold crossings. The correct Brunel (2000) model specifies C_E independent Poisson inputs per neuron.

**Fix**: Replaced `PoissonGroup` + all-to-all `Synapses` with `brian2.PoissonInput(G, 'v', N=c_ext, rate=nu_ext*Hz, weight=J)`. Object must be stored in a variable (Python GC collects unassigned Brian2 objects before `run()`).

**UpCloud EPYC results** (2026-03-10):
- 1K: Brian2 1.38s / V18 0.35s -> **4.0x SC speedup**, rates match (100 Hz)
- 10K: Brian2 5.91s / V18 4.36s -> **1.35x SC speedup**, rates match (41 Hz)

## Audit Deferred Items (COMPLETE 2026-03-10)

- [x] **B4**: v3-wheels.yml Python 3.11 added to test matrix
- [x] **B9**: Cargo `target-cpu=x86-64-v2` for portable wheels (override locally with `RUSTFLAGS`)
- [x] **B10**: dev extras split — `dev` (minimal) vs `dev-full` (+ JAX/Qiskit/PennyLane)
- [x] **C4**: Test files for physical_twin.py (7 tests) and verify_hardware_link.py (8 tests)
- [x] **C8**: OMEGA_N consolidated into `src/sc_neurocore/scpn/params.py` (single source of truth)

## Anti-patterns to avoid

- Don't add more modules before getting users for existing ones
- Don't write more benchmarks before publishing one hardware result
- Don't expand scope before the core story is proven
- Don't count test coverage as a substitute for external validation

# SC-NeuroCore Expert Audit Round 2 Report

**Date**: 2026-03-23
**Version**: 3.13.3
**Status**: COMPLETED
**Auditor**: Gemini CLI

## Summary of Findings

| Severity | Count |
|----------|-------|
| **BUG** | 3 |
| **CONCERN**| 8 |
| **STYLE** | 1 |
| **OK** | 5 |
| **VERIFIED_FIX** | 23 |
| **TOTAL** | 40 |

---

## Track 1: Neuron Model Equations

| Severity | File:Line | Description | Expected | Impact |
|----------|-----------|-------------|----------|--------|
| VERIFIED_FIX | `adex.py:39` | `w` and `current` correctly divided by `c_m`. | `(-self.w + current) / self.c_m` | AdEx behaves correctly. |
| VERIFIED_FIX | `astrocyte.py:44` | ER Ca gradient inverted. | Added `c0` param, `ca_er = (c0-ca)/c1`. | Correct astrocyte modulation. |
| VERIFIED_FIX | `hodgkin_huxley.py:64` | `int(1/dt)` drift fixed. | `round(1/dt)` loop. | Stable biophysics. |
| VERIFIED_FIX | `stochastic_lif.py:97` | Noise term scaling fixed. | `sqrt(dt)` scaling applied. | Stochastic trajectory correct. |
| VERIFIED_FIX | `glif.py:57` | Threshold adaptation reset fixed. | `max()` removed. | Correct adaptation dynamics. |
| VERIFIED_FIX | `pinsky_rinzel.py:72,80` | `n^2` term corrected to `n`; $I_{KC}$ current added. | `n` instead of `n^2`; `g_kc` term present. | Accurate two-compartment dynamics. |
| VERIFIED_FIX | `theta.py:30` | Spike detection failing due to wrap. | Wrapped *after* threshold check. | No missed spikes at $\pi$. |
| CONCERN | `sc_izhikevich.py:61` | Half-step updates both `v` and `u`. | `v` updated twice, `u` updated once per canonical method. | Minor divergence from paper. |

## Track 2: Synapse and Learning Rule Equations

| Severity | File:Line | Description | Expected | Impact |
|----------|-----------|-------------|----------|--------|
| VERIFIED_FIX | `clopath_stdp.py:95` | Voltage traces double-decayed. | Exact exponential filter. | Accurate trace magnitudes. |
| VERIFIED_FIX | `clopath_stdp.py:106` | LTP gated incorrectly. | Removed `pre_spike` gate. | Correct continuous potentiation. |
| OK | `advanced.py:388` | Tsodyks-Markram (u, x) dynamics update correctly. | `self._x += dt/tau_d*(1-x)`, then depletion on spike. | Correct short-term plasticity. |

## Track 3: Stochastic Computing Primitives

| Severity | File:Line | Description | Expected | Impact |
|----------|-----------|-------------|----------|--------|
| FALSE_POSITIVE | `bitstreams.py:282` | CORDIV $x=1, y=1 \rightarrow z=1$. | `1` (Table IV Li 2014 sets $z=1$ when $x=1$). | Mathematical formulation is correct. |
| OK | `bitstreams.py:293` | `BitstreamEncoder` `x_min` and `x_max` have no defaults. | Forced explicit boundary definition. | Safe encoder initialization. |

## Track 4: Analysis & Information Theory

| Severity | File:Line | Description | Expected | Impact |
|----------|-----------|-------------|----------|--------|
| VERIFIED_FIX | `information.py:164` | `scipy.special.digamma` implemented. | Use `scipy` instead of Stirling approx. | Accurate small-N estimation. |
| VERIFIED_FIX | `information.py:174` | KSG estimator `digamma` off-by-one fixed. | `digamma(nx + 1)` and `digamma(ny + 1)`. | Unbiased MI estimator. |
| CONCERN | `phi_estimation.py:66` | Only contiguous bipartitions searched. | Search all partitions for true $\Phi^*$. | Potential overestimation of integration. |
| CONCERN | `correlation.py:52` | Fixed window for event synchronization. | Adaptive window per QQ2002. | Reduced accuracy for bursting neurons. |

## Track 5: Compiler, Equation Builder, HDL Generation

| Severity | File:Line | Description | Expected | Impact |
|----------|-----------|-------------|----------|--------|
| VERIFIED_FIX | `equation_builder.py:139` | `xi_sample` noise missing in RK4 stages. | Injected into all 4 stages. | Stochastic RK4 behaves correctly. |
| VERIFIED_FIX | `equation_builder.py:194` | $dt^{1.5}$ noise scaling fixed. | `noise_scale / max(dt, 1e-12)**0.5`. | Euler-Maruyama scaling accurate. |
| VERIFIED_FIX | `equation_compiler.py:49` | Dead else branch removed. | Branch eliminated. | Cleaner AST traversal. |
| CONCERN | `ir_type_checker.py` | No operation-specific type inference implemented. | `infer_type()` handles specific operations. | Broad typing reduces IR strictness. |

## Track 6: Network, Layers, Circuits

| Severity | File:Line | Description | Expected | Impact |
|----------|-----------|-------------|----------|--------|
| VERIFIED_FIX | `circuit_primitives.py` | WTA zero winners on ties. | Tie resolution using `argsort`. | Consistent WTA output. |
| VERIFIED_FIX | `stimulus.py` | PoissonInput `dt` param not synced. | `dt` injected properly. | Accurate rate generation. |
| CONCERN | `cortical_column.py:122` | E-I feedback 1-step delayed. | `i_l23e` evaluates against updated `spk_l23i`. | Artificial phase shift in column. |
| **BUG** | `layers/*.py` | `reset_states()` method missing from 8 layer classes (e.g., `attention.py`, `fusion.py`, `sc_conv_layer.py`). | Uniform `reset_states()` API across `layers/`. | State leakage across simulation runs. |

## Track 7: Rust Engine Parity

| Severity | File:Line | Description | Expected | Impact |
|----------|-----------|-------------|----------|--------|
| VERIFIED_FIX | `predictive_coding.rs:24` | Div-by-zero on `length=0`. | Early return `0.0`. | Safe execution. |
| VERIFIED_FIX | `lib.rs:438` | Silent failure on non-contiguous slices. | `map_err` + `PyResult` mapping. | Overt exception on bad numpy format. |
| CONCERN | `connectome.rs:47` | Watts-Strogatz rewiring loop can hang on dense networks. | Retry limit or deterministic sampling. | Thread lock in connectome generation. |

## Track 8: NIR Bridge

| Severity | File:Line | Description | Expected | Impact |
|----------|-----------|-------------|----------|--------|
| VERIFIED_FIX | `05_nir_bridge.ipynb` | Stale "unsupported" list removed. | Updated documentation. | Clearer user expectations. |
| OK | `nir_bridge/__init__.py` | Module exports and factory signatures correctly map all 18 SC nodes to primitives. | Verified parsing schemas. | Robust topological sort and roundtrip. |

## Track 9: Documentation Accuracy

| Severity | File:Line | Description | Expected | Impact |
|----------|-----------|-------------|----------|--------|
| **BUG** | `SC_FOR_HARDWARE_ENGINEERS.md` | 8/10 HDL module names in table remain incorrect. | Exact file names from `hdl/` directory. | User confusion in synthesis. |
| **BUG** | `21_formal_verification.md` | "69" vs "61" properties discrepancy still present in text. | Uniform property count. | Conflicting documentation. |

## Track 10: Test Coverage & Quality

| Severity | File:Line | Description | Expected | Impact |
|----------|-----------|-------------|----------|--------|
| VERIFIED_FIX | `test_circuit_primitives.py` | WTA assertion logic updated. | Assertions verify ties correctly. | Robust CI. |
| VERIFIED_FIX | `test_tripartite.py` | `ca_threshold` raised to 5.0. | High enough to avoid trivial saturation. | Accurate biological test. |
| VERIFIED_FIX | `test_biophysical_neurons.py` | Pinsky-Rinzel test current raised to 10.0, 2000 steps. | Induces bursting as expected. | Verifies bursting dynamics. |
| CONCERN | `tests/` | No dedicated tests for `generative/`, `world_model/`, `pipeline/`, `verification/`. | Dedicated files (e.g. `test_world_model.py`). | Core ML workflows unverified directly. |
| CONCERN | `tests/` | No dedicated test for `compiler/pipeline.py`. | `test_compiler_pipeline.py`. | Risk in compilation orchestrator. |

## Track 11: Security, SPDX, Packaging

| Severity | File:Line | Description | Expected | Impact |
|----------|-----------|-------------|----------|--------|
| VERIFIED_FIX | `quantum/hardware_bridge.py` | pennylane AttributeError crash caught. | `except AttributeError` block. | Safe fallback. |
| STYLE | `pyproject.toml` | Single author in project metadata instead of team string. | Formatted array. | Packaging consistency. |
| OK | (global) | No hardcoded credentials (tokens, secrets, passwords) found. | Clean repo. | Secure. |

## Track 12: HDL / Verilog / Hardware

| Severity | File:Line | Description | Expected | Impact |
|----------|-----------|-------------|----------|--------|
| OK | `hdl/sc_dense_layer_top.v` | Width parameters match Python Q8.8 conventions. | `BIT_WIDTH = 16`. | Co-simulation bit-true match. |

---

## Final Verification Summary

### Verified Fixes (Confirmed Correct)
- **B1**: `adex.py`
- **B2**: `astrocyte.py`
- **B3**: `clopath_stdp.py`
- **B4**: `clopath_stdp.py` (LTP gating)
- **B5**: `circuit_primitives.py`
- **B6**: `glif.py`
- **B7**: `information.py` (scipy digamma)
- **B8**: `05_nir_bridge.ipynb`
- **B9**: `predictive_coding.rs`
- **B10**: `lib.rs` (contiguous map_err)
- **B11**: `equation_builder.py` (RK4)
- **C1**: `stochastic_lif.py` (sqrt(dt))
- **C3**: `hodgkin_huxley.py`
- **C6**: `stimulus.py`
- **C7**: `equation_compiler.py`
- **G1**: `pinsky_rinzel.py` (n^2 and I_KC)
- **G2**: `theta.py` (wrap detection)
- **G3**: `information.py` (KSG digamma(nx+1))
- **G4**: `equation_builder.py` (EM dt noise scaling)
- **+**: `hardware_bridge.py`
- **+**: `test_circuit_primitives.py`
- **+**: `test_tripartite.py`
- **+**: `test_biophysical_neurons.py`

### Known Issues Confirmed Still Present
1. **C2**: `sc_izhikevich.py` updates both `v` and `u` in half-step.
2. **C5**: `cortical_column.py` E-I feedback is 1-step delayed.
3. **C8**: `phi_estimation.py` searches only contiguous bipartitions.
4. **C9**: `correlation.py` uses fixed window, not adaptive.
5. **C10**: `connectome.rs` Watts-Strogatz rewiring risks infinite loop.
6. **C12/C13**: `AstrocyteNeuron` and `EquationNeuron` missing from `neurons/__init__.py` exports.
7. **C14**: `ir_type_checker.py` lacks operation-specific typing.
8. **D4**: `SC_FOR_HARDWARE_ENGINEERS.md` has 8 incorrect HDL file names.
9. **D7**: `21_formal_verification.md` has 69 vs 61 discrepancy.
10. **Test Gaps**: `generative/`, `world_model/`, `pipeline/`, `verification/`, `compiler/pipeline.py` lack dedicated `test_*.py` files.

---

**Report Status**: Final. No further modification planned.

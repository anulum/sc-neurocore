<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SC-NeuroCore — GranuleCell model reference -->

# GranuleCell

**Module:** `src/sc_neurocore/neurons/models/granule_cell.py`<br>
**Rust engine:** `engine/src/neurons/cerebellar/granule.rs`<br>
**Reference:** D'Angelo et al., *Journal of Neuroscience* 21(3), 2001<br>
**Family:** conductance-based cerebellar granule-cell model with tonic GABA

---

## Biological context

Cerebellar granule cells are small, high-input-resistance neurons receiving
mossy-fibre excitation inside glomeruli while Golgi-cell axons provide tonic
GABAergic inhibition. Their intrinsic dynamics combine fast sodium spiking,
delayed and A-type potassium recovery, T-type calcium rebound, calcium-activated
potassium after-hyperpolarisation, HCN sag current, leak, and tonic GABA.

SC-NeuroCore models that surface as a sub-stepped Hodgkin-Huxley-type system,
not as a reset-only leaky integrate-and-fire approximation.

---

## State variables

| Variable | Meaning |
|----------|---------|
| `v` | Membrane potential in mV |
| `m`, `h` | Sodium activation and inactivation |
| `n` | Delayed-rectifier potassium activation |
| `a`, `b` | A-type potassium activation and inactivation |
| `m_t`, `s` | T-type calcium activation and inactivation |
| `ca` | Intracellular calcium proxy |
| `r` | HCN activation |

---

## Current balance

The membrane update integrates:

$$
C_m\frac{dV}{dt} =
-(I_{Na} + I_{Kdr} + I_{KA} + I_{CaT} + I_{KCa} + I_h + I_L + I_{GABA})
+ g\,I_{ext}
$$

with the maintained current definitions:

| Current | Definition |
|---------|------------|
| Sodium | \(I_{Na}=g_{Na}m^3h(V-E_{Na})\) |
| Delayed rectifier | \(I_{Kdr}=g_{Kdr}n^4(V-E_K)\) |
| A-type potassium | \(I_{KA}=g_{KA}a^3b(V-E_K)\) |
| T-type calcium | \(I_{CaT}=g_Tm_T^2s(V-E_{Ca})\) |
| Calcium-activated potassium | \(I_{KCa}=g_{KCa}\frac{Ca^2}{Ca^2+K_d^2}(V-E_K)\) |
| HCN | \(I_h=g_hr(V-E_h)\) |
| Leak | \(I_L=g_L(V-E_L)\) |
| Tonic GABA | \(I_{GABA}=g_{tonic}(V-E_{GABA})\) |

Gates use overflow-stable Boltzmann steady states and closed-form first-order
relaxation. The membrane potential advances with the exact ohmic conductance
solution over each voltage-frozen sub-step. The default Python/Rust/Go/Julia
paths use four sub-steps per model step.

---

## Default parameters

| Parameter | Value | Meaning |
|-----------|------:|---------|
| `c_m` | 1.0 | Membrane capacitance |
| `g_na` | 17.0 | Sodium conductance |
| `g_kdr` | 9.0 | Delayed-rectifier potassium conductance |
| `g_ka` | 1.0 | A-type potassium conductance |
| `g_t` | 0.5 | T-type calcium conductance |
| `g_kca` | 3.5 | Calcium-activated potassium conductance |
| `g_h` | 0.03 | HCN conductance |
| `g_l` | 0.1 | Leak conductance |
| `g_tonic` | 0.2 | Tonic GABA conductance |
| `e_na` | 87.4 | Sodium reversal |
| `e_k` | -84.7 | Potassium reversal |
| `e_ca` | 129.3 | Calcium reversal |
| `e_h` | -40.0 | HCN reversal |
| `e_l` | -58.0 | Leak reversal |
| `e_gaba` | -75.0 | GABA reversal |
| `tau_ca` | 10.0 | Calcium decay time constant |
| `kd_kca` | 0.2 | KCa half-saturation |
| `dt` | 0.5 | Model timestep |
| `sub_steps` | 4 | Integration sub-steps |
| `gain` | 1.0 | External-current gain |

---

## Safety and fidelity contract

The maintained Python, Rust engine, Go service, Julia kernel, and Rust safety
shim now share the same runtime boundary:

- finite state and parameter validation before integration
- physical voltage bounds before integration
- gates constrained to `[0, 1]`
- non-negative calcium and conductances
- positive capacitance, calcium constants, timestep, and sub-step count
- finite runtime input current
- exact gate, calcium, and conductance-form voltage integration with local
  candidate state and no partial mutation on invalid input
- spike reporting on upward crossing of 0 mV

This keeps optional acceleration surfaces aligned with the production dynamics
instead of returning placeholders or normalising corrupted state silently.

---

## Verification surfaces

Module-specific tests cover the Python model and Go service. The Rust engine and
Rust safety shim carry granule-specific tests in their respective crates/files,
and the Julia kernel is covered by the same closed-form kinetics parity
assertion used during release verification. The checks exercise bounded state
evolution, D'Angelo current-surface presence, closed-form gate/calcium kinetics,
tonic GABA suppression, T-type de-inactivation at rest, invalid-configuration
rejection, non-finite drive preservation, and corrupted-state preservation.

---

## Benchmark status

Remeasured locally with Criterion on 2026-05-31 after exact gate, calcium, and
conductance-form voltage integration:

| Benchmark | Median |
|-----------|-------:|
| `granule_10k_steps` | 7.64 ms |
| Per step | **0.764 µs** |

Artifact:
`benchmarks/results/local_i5_11600k_criterion_2026-05-31_granule_cell.json`.

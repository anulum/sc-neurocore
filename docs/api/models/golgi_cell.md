<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- (C) Concepts 1996-2026 Miroslav Sotek. All rights reserved. -->
<!-- (C) Code 2020-2026 Miroslav Sotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SC-NeuroCore GolgiCell model reference -->

# GolgiCell

**Module:** `engine/src/neurons/cerebellar.rs`
**Reference:** Solinas et al., Front Cell Neurosci 1:2, 2007
**Family:** Full Hodgkin-Huxley with 11 ionic currents, exact first-order gates, exact calcium relaxation, and conductance-form membrane integration
**State variables:** `v`, `m`, `h`, `p_na`, `n`, `a`, `b`, `w`, `m_t`, `s`, `c_n`, `r`, `ca`

---

## Biological Context

Golgi cells are large inhibitory interneurons in the cerebellar granular layer. They provide tonic and phasic GABAergic/glycinergic inhibition to granule cells at glomeruli, forming feedforward and feedback inhibitory loops that control granule cell excitability and temporal filtering of mossy fibre input.

Key features:
- **Spontaneous pacemaker firing** (3-10 Hz): driven by persistent Na (INaP) + Ih + depolarised leak
- **11 ionic currents** (Solinas 2007): Na_t, Na_p, K_dr, K_A, K_M, Ca_T, Ca_N, BK, SK, Ih, leak
- **Two Ca channels**: T-type (low-voltage, rebound) and N-type (high-voltage, AHP trigger)
- **BK + SK afterhyperpolarisation**: BK (fast AHP, V+Ca dependent) + SK (slow AHP, Ca dependent) shape spike frequency adaptation
- **Ih (HCN)**: sag current, contributes to resting potential and pacemaker
- **K_M (KCNQ)**: slow muscarinic K+, limits high-frequency firing

---

## Equations

$$C_m \frac{dV}{dt} = -(I_{Na_t} + I_{Na_p} + I_{K_{dr}} + I_{K_A} + I_{K_M} + I_{Ca_T} + I_{Ca_N} + I_{BK} + I_{SK} + I_h + I_L) + I_{ext}$$

### Exact numerical update

Each runtime advances every first-order gate with the closed-form relaxation solution for the fixed sub-step voltage, updates Ca2+ with the exact linear relaxation toward inward Ca-current entry, and advances membrane voltage with the exact conductance-form solution for fixed sub-step conductances. Candidate state is committed only when voltage, gates, calcium, conductances, capacitance, calcium constants, timestep, sub-step count, and gain remain finite and physiological; invalid or excess current returns no spike and preserves pre-step state.

### Ionic Currents

| Current | Gating | Formula |
|---------|--------|---------|
| INa_t | m³h | $g_{Na_t} \cdot m^3 h \cdot (V - E_{Na})$ |
| INa_p | p | $g_{Na_p} \cdot p_{Na} \cdot (V - E_{Na})$ |
| IK_dr | n⁴ | $g_{K_{dr}} \cdot n^4 \cdot (V - E_K)$ |
| IK_A | a³b | $g_{K_A} \cdot a^3 b \cdot (V - E_K)$ |
| IK_M | w | $g_{K_M} \cdot w \cdot (V - E_K)$ |
| ICa_T | m_t²s | $g_{Ca_T} \cdot m_t^2 s \cdot (V - E_{Ca})$ |
| ICa_N | c² | $g_{Ca_N} \cdot c_n^2 \cdot (V - E_{Ca})$ |
| IBK | V+Ca Boltzmann | $g_{BK} \cdot f_{BK}(V, [Ca]) \cdot (V - E_K)$ |
| ISK | Hill n=2 | $g_{SK} \cdot \frac{[Ca]^2}{[Ca]^2 + K_d^2} \cdot (V - E_K)$ |
| Ih | r | $g_h \cdot r \cdot (V - E_h)$ |
| IL | ohmic | $g_L \cdot (V - E_L)$ |

### BK Ca-voltage dependence

$$V_{1/2}^{BK} = 100 - 120 \cdot \frac{[Ca]^2}{[Ca]^2 + K_d^2}$$

$$f_{BK} = \frac{1}{1 + \exp(-(V - V_{1/2}^{BK})/15)}$$

### Ca²⁺ dynamics

$$\frac{d[Ca]}{dt} = -\frac{(I_{Ca_T} + I_{Ca_N})_{inward} \cdot 0.001}{\text{volume factor}} - \frac{[Ca]}{\tau_{Ca}}$$

---

## Parameters

| Parameter | Value | Unit | Description |
|-----------|-------|------|-------------|
| `g_na_t` | 48.0 | mS/cm² | Transient Na (Solinas Table 1) |
| `g_na_p` | 0.2 | mS/cm² | Persistent Na (pacemaker) |
| `g_kdr` | 16.0 | mS/cm² | Delayed rectifier K |
| `g_ka` | 8.0 | mS/cm² | A-type K (V1/2=-27 mV) |
| `g_km` | 1.0 | mS/cm² | Muscarinic K (KCNQ) |
| `g_cat` | 0.5 | mS/cm² | T-type Ca²⁺ |
| `g_can` | 1.0 | mS/cm² | N-type Ca²⁺ |
| `g_bk` | 3.0 | mS/cm² | BK (fast AHP) |
| `g_sk` | 1.0 | mS/cm² | SK (slow AHP) |
| `g_h` | 0.1 | mS/cm² | Ih (HCN) |
| `g_l` | 0.05 | mS/cm² | Leak |
| `e_na` | 55.0 | mV | Na reversal |
| `e_k` | -90.0 | mV | K reversal |
| `e_ca` | 120.0 | mV | Ca²⁺ reversal |
| `e_h` | -40.0 | mV | Ih reversal |
| `e_l` | -55.0 | mV | Leak reversal (depolarised, pacemaker) |
| `tau_ca` | 200.0 | ms | Ca²⁺ decay time constant |
| `kd_bk` | 1.0 | µM | BK Ca²⁺ Kd (Hill n=2) |
| `kd_sk` | 0.5 | µM | SK Ca²⁺ Kd (Hill n=2) |
| `dt` | 0.5 | ms | Integration timestep (10 sub-steps) |

---

## Pipeline Status

| Checklist | Status |
|-----------|--------|
| Rust implementation | `engine/src/neurons/cerebellar.rs` |
| PyO3 wrapper | `pyo3_neurons.rs` (state: v, m, h, p_na, n, a, b, w, m_t, s, c_n, r, ca) |
| NetworkRunner wired | `NeuronVariant::Golgi` |
| `create_neuron("GolgiCell")` | Yes |
| `supported_models()` | Includes "GolgiCell" |
| Behavior tests | Rust engine 21; Python model 4; Go service 3; Rust safety 5 |
| Benchmark | `golgi_1k_steps`: **2.96 ms** median (2.96 µs/step), i5-11600K |

---

## Benchmark (Criterion, i5-11600K @ 3.90 GHz)

| Benchmark | Median |
|-----------|-------:|
| golgi_1k_steps | 2.96 ms |
| Per step | **2.96 µs** |

11 ionic currents, 11 first-order gates, Ca2+ relaxation, and 10 exact sub-steps (dt_sub=0.05 ms). Measured 2026-05-31 on local i5-11600K.

---

## Findings

1. **Full Solinas 2007 model.** All 11 currents implemented with publication-matched kinetics.
2. **Fires with excitatory input.** Sustained spiking with I=15 µA/cm². Verified.
3. **Near-spontaneous firing.** Fires with minimal input (0.5 µA/cm²) due to NaP + depolarised leak. Verified.
4. **Persistent Na contributes to pacemaking.** Removing NaP reduces excitability. Verified.
5. **K_M limits high-frequency firing.** Removing K_M increases spike rate. Verified.
6. **K_A modulates firing pattern.** Transient A-type current affects spike rate. Verified.
7. **Ih produces sag.** Hyperpolarisation with Ih → less negative V than without. Verified.
8. **BK fast AHP.** BK channels activate during spikes, contribute to repolarisation. Verified.
9. **SK slow adaptation.** Removing SK increases firing rate (Ca²⁺-dependent slow AHP). Verified.
10. **Ca²⁺ accumulates during spiking.** Ca²⁺ entry via Ca_T + Ca_N. Verified.
11. **All gates bounded [0,1].** 11 gating variables + Ca2+ non-negative. Verified.
12. **Invalid-input fail-closed behaviour.** NaN, infinite, excess-current, and corrupted-state paths preserve pre-step state and return no spike. Verified.

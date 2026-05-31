<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved. -->
<!-- Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SC-NeuroCore - DCNNeuron model documentation -->

# DCNNeuron

**Module:** `engine/src/neurons/cerebellar.rs`
**Reference:** Llinás & Mühlethaler, J Physiol 404:241, 1988; Jahnsen, J Physiol 372:129, 1986
**Family:** WB Na+/K+ + persistent Na + T-type Ca²⁺ + Ca²⁺-AHP + Ih
**State variables:** `v`, `h`, `n`, `p` (NaP), `s` (T-type inact), `r` (Ih), `ca`

---

## Biological Context

Deep cerebellar nuclei (DCN) neurons are the main output neurons of the cerebellum. They receive massive GABAergic inhibition from Purkinje cells and excitatory input from mossy fibre and climbing fibre collaterals. DCN neurons relay cerebellar computations to the thalamus, brainstem, and spinal cord.

Key features:
- **7 ionic currents**: INa_t, INaP, IK_dr, ICa_T, IAHP, Ih, IL
- **Rebound bursting**: T-type Ca²⁺ de-inactivates during Purkinje inhibition → burst on release
- **Persistent Na (INaP)**: amplifies subthreshold depolarisation, contributes to spontaneous activity
- **Ca²⁺-dependent AHP**: limits burst duration and sustained firing rate (Hill n=2)
- **Ih (HCN)**: sag current, pacemaker contribution
- **Spontaneous firing**: INaP + Ih + depolarised leak drive autonomous activity (~10-50 Hz)

---

## Equations

$$C_m \frac{dV}{dt} = -(I_{Na_t} + I_{Na_p} + I_{K_{dr}} + I_{Ca_T} + I_{AHP} + I_h + I_L) + I_{ext}$$

| Current | Gating | Formula |
|---------|--------|---------|
| INa_t | m³h | $g_{Na} \cdot m_\infty^3 h \cdot (V - E_{Na})$ |
| INaP | p | $g_{NaP} \cdot p \cdot (V - E_{Na})$ |
| IK_dr | n⁴ | $g_K \cdot n^4 \cdot (V - E_K)$ |
| ICa_T | m_t²s | $g_T \cdot m_{T,\infty}^2 s \cdot (V - E_{Ca})$ |
| IAHP | Hill | $g_{AHP} \cdot \frac{[Ca]^2}{[Ca]^2 + K_d^2} \cdot (V - E_K)$ |
| Ih | r | $g_h \cdot r \cdot (V - E_h)$ |
| IL | ohmic | $g_L \cdot (V - E_L)$ |

### NaP gating (Boltzmann)

$$p_\infty = \frac{1}{1 + \exp(-(V+48)/5)}$$

### Ca²⁺ dynamics

$$\frac{d[Ca]}{dt} = -(I_{Ca_T})_{inward} \cdot 0.001 - \frac{[Ca]}{\tau_{Ca}} + 0.5 \text{ on spike}$$

---

## Parameters

| Parameter | Value | Unit | Description |
|-----------|-------|------|-------------|
| `g_na` | 35.0 | mS/cm² | Transient Na |
| `g_nap` | 0.5 | mS/cm² | Persistent Na |
| `g_k` | 9.0 | mS/cm² | Delayed rectifier K |
| `g_t` | 0.1 | mS/cm² | T-type Ca²⁺ |
| `g_ahp` | 2.0 | mS/cm² | Ca²⁺-dependent AHP |
| `g_h` | 0.02 | mS/cm² | Ih (HCN) |
| `g_l` | 0.2 | mS/cm² | Leak |
| `tau_ca` | 150.0 | ms | Ca²⁺ decay |
| `kd_ahp` | 0.5 | µM | AHP Ca²⁺ Kd (Hill n=2) |
| `dt` | 0.5 | ms | Timestep (20 sub-steps) |

## Runtime Safety Contract

The maintained Python, Rust-engine, Go, Julia, and Rust safety surfaces now use
the same seven-current sub-stepped update. Each step validates finite state,
finite input current, gate bounds, non-negative conductances, positive
capacitance, positive calcium and timestep constants, and non-negative gain
before integration. Candidate state is computed locally and committed only when
all state variables remain finite; invalid runtime input or corrupted state
preserves the previous state on non-throwing runtimes, while the Python surface
raises before mutation.

---

## Pipeline Status

| Checklist | Status |
|-----------|--------|
| Rust implementation | `engine/src/neurons/cerebellar.rs` |
| PyO3 wrapper | `pyo3_neurons.rs` (state: v, h, n, p, s, r, ca) |
| NetworkRunner wired | `NeuronVariant::DCN` |
| `create_neuron("DCNNeuron")` | Yes |
| `supported_models()` | Includes "DCNNeuron" |
| coverage tests | Rust engine tests plus module-specific Python and Go tests for seven-current surface, gate/Ca²⁺ bounds, T-type de-inactivation, Ih depolarisation, invalid configuration rejection, non-finite input preservation, and invalid candidate preservation |
| Benchmark | `dcn_1k_steps`: **2.14 ms** (2.14 µs/step), i5-11600K |

---

## Benchmark (Criterion, i5-11600K @ 3.90 GHz)

| Benchmark | Median |
|-----------|-------:|
| dcn_1k_steps | 2.14 ms |
| Per step | **2.14 µs** |

7 currents, 20 sub-steps (dt_sub=0.025 ms). Historical timing was measured
2026-04-05; regenerate before using as release evidence.

---

## Findings

1. **7 ionic currents.** Na_t, Na_p, K_dr, Ca_T, AHP, Ih, leak — resolves MODERATE audit finding.
2. **Fires with excitatory input.** Sustained spiking with I=5 µA/cm². Verified.
3. **Spontaneous activity.** INaP + Ih drive autonomous firing without input. Verified.
4. **Rebound burst via T-type.** De-inactivated T-type facilitates firing after hyperpolarisation. Verified.
5. **INaP increases excitability.** Removing INaP reduces firing rate. Verified.
6. **AHP limits firing rate.** Removing AHP increases sustained rate. Verified.
7. **Ca²⁺ accumulates during spiking.** Entry via T-type + spike-triggered bolus. Verified.
8. **Ih depolarises from hyperpolarised state.** Sag current confirmed. Verified.

<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SC-NeuroCore — StellateCell model reference -->

# StellateCell

**Python module:** `src/sc_neurocore/neurons/models/stellate_cell.py`<br>
**Rust engine:** `engine/src/neurons/cerebellar/stellate.rs`<br>
**Reference:** Sultan & Bower, *Journal of Comparative Neurology* 409:63,
1999; Häusser & Clark, *Neuron* 19:665, 1997<br>
**Family:** Wang-Buzsáki sodium/potassium core with Kv3.1 activation

---

## Biological context

Cerebellar stellate cells are inhibitory molecular-layer interneurons receiving
parallel-fibre excitation and projecting to distal Purkinje-cell dendrites. The
maintained model represents the fast-spiking interneuron phenotype with a
Wang-Buzsáki sodium/potassium membrane core plus a Kv3.1 potassium current for
rapid action-potential repolarisation.

---

## State variables

| Variable | Meaning |
|----------|---------|
| `v` | Membrane potential in mV |
| `h` | Sodium inactivation gate |
| `n` | Delayed-rectifier potassium activation gate |
| `p` | Kv3.1 activation gate |

---

## Current balance

The membrane equation is:

$$
C_m\frac{dV}{dt} = -I_{Na} - I_K - I_{Kv3} - I_L + gI_{ext}
$$

with:

| Current | Definition |
|---------|------------|
| Sodium | \(I_{Na}=g_{Na}m_\infty^3h(V-E_{Na})\) |
| Delayed-rectifier potassium | \(I_K=g_Kn^4(V-E_K)\) |
| Kv3.1 potassium | \(I_{Kv3}=g_{Kv3}p^2(V-E_K)\) |
| Leak | \(I_L=g_L(V-E_L)\) |

The sodium activation gate uses the Wang-Buzsáki steady-state activation rate.
The `h` and `n` gates use Wang-Buzsáki alpha/beta kinetics scaled by `phi`.
The Kv3.1 `p` gate evolves on its own voltage-dependent time constant:

$$
p_\infty(V)=\frac{1}{1+\exp(-(V+10)/10)}
$$

$$
\tau_p(V)=1+\frac{4}{1+\exp((V+20)/15)}
$$

---

## Default parameters

| Parameter | Value | Meaning |
|-----------|------:|---------|
| `g_na` | 35.0 | Sodium conductance |
| `g_k` | 9.0 | Delayed-rectifier potassium conductance |
| `g_kv3` | 3.0 | Kv3.1 potassium conductance |
| `g_l` | 0.1 | Leak conductance |
| `e_na` | 55.0 | Sodium reversal |
| `e_k` | -90.0 | Potassium reversal |
| `e_l` | -65.0 | Leak reversal |
| `c_m` | 0.5 | Reduced capacitance for small stellate-cell soma |
| `phi` | 5.0 | Wang-Buzsáki gating-rate scale |
| `dt` | 0.5 | Model timestep |
| `v_threshold` | -20.0 | Spike reporting threshold |
| `gain` | 1.0 | External-current gain |
| `sub_steps` | 50 | Integration sub-steps on maintained acceleration paths |

---

## Safety and fidelity contract

The Python model, Rust engine, Go service, Julia kernel, and Rust safety shim
share the same operational boundary:

- finite state and parameter validation before integration
- gates constrained to `[0, 1]`
- non-negative conductances
- positive capacitance, rate scale, timestep, and sub-step count
- membrane voltage constrained to the physical `[-100, 60] mV` operating range
- finite runtime input current
- overflow-bounded exponentials in rate and Boltzmann calculations
- exact closed-form relaxation for Wang-Buzsáki `h` and `n` gates
- exact first-order relaxation for the Kv3.1 `p` gate
- exact conductance-form membrane integration over each voltage-frozen sub-step
- local candidate-state integration with no partial mutation on invalid input
- spike reporting when the membrane candidate crosses `v_threshold`, followed
  by reset to `-65 mV`

This prevents optional acceleration paths from returning placeholders,
swallowing numerical errors, or silently normalising corrupted state.

---

## Verification surfaces

Module-specific Python tests cover bounded state evolution, Kv3.1 activation
under depolarisation, invalid-configuration rejection, non-finite-drive
preservation, corrupted-state preservation, and closed-form gate kinetics. The
Go service, Julia kernel, Rust safety shim, and Rust engine carry matching
module-owned checks for the exact gate-relaxation contract and fail-closed
runtime guards.

---

## Benchmark status

The maintained Rust engine path was remeasured on 2026-05-31 after the exact
gate and conductance integration hardening:

- command: `cargo bench --manifest-path engine/Cargo.toml --bench full_bench stellate_1k_steps`
- hardware: Intel Core i5-11600K @ 3.90 GHz, 6C/12T, verified with `lscpu`
- Criterion median: `6.075846375 ms` per 1k steps (`6.075846375 µs` per step)
- artifact:
  `benchmarks/results/local_i5_11600k_criterion_2026-05-31_stellate_cell.json`

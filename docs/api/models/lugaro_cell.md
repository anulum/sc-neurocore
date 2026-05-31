<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SC-NeuroCore — LugaroCell model reference -->

# LugaroCell

**Python module:** `src/sc_neurocore/neurons/models/lugaro_cell.py`<br>
**Rust engine:** `engine/src/neurons/cerebellar.rs`<br>
**Reference:** Dieudonné & Bhatt, *Journal of Physiology* 548:97, 2003;
Lainé & Bhatt, *Frontiers in Systems Neuroscience* 1:4, 2007<br>
**Family:** leaky integrate-and-fire model with adaptation and serotonin gain

---

## Biological context

Lugaro cells are rare fusiform inhibitory interneurons in the upper granular
layer. They project horizontally and inhibit Golgi cells and molecular-layer
interneurons. Their depolarised resting potential and serotonergic sensitivity
make them a circuit-level source of state-dependent disinhibition.

The maintained SC-NeuroCore model intentionally uses a compact LIF-adaptation
surface because the project audit records Lugaro as fidelity-acceptable without
a detailed Hodgkin-Huxley model requirement.

---

## State variables

| Variable | Meaning |
|----------|---------|
| `v` | Membrane potential in mV |
| `adapt` | Non-negative hyperpolarising adaptation current |

---

## Dynamics

The effective serotonergic gain is:

$$
g_{eff}=g(1+0.5s)
$$

where \(s \in [0,1]\) is the serotonin level. The membrane target under constant
drive over one step is:

$$
V_\infty = V_{rest} + g_{eff}I_{ext} - adapt
$$

The maintained paths use closed-form first-order relaxation instead of a raw
Euler increment:

$$
V_{next}=V_\infty+(V-V_\infty)\exp(-dt/\tau_m)
$$

The adaptation target is non-negative:

$$
adapt_\infty = \max(0, a_{adapt}\max(0,V_{next}-V_{rest}))
$$

and the candidate adaptation state is:

$$
adapt_{next}=adapt_\infty+(adapt-adapt_\infty)\exp(-dt/\tau_{adapt})
$$

If \(V_{next}\) crosses threshold, the model reports a spike, resets voltage to
`v_reset`, and adds the spike-triggered adaptation increment.

---

## Default parameters

| Parameter | Value | Meaning |
|-----------|------:|---------|
| `v_rest` | -55.0 | Depolarised resting potential |
| `v_reset` | -65.0 | Post-spike reset potential |
| `v_threshold` | -48.0 | Spike threshold |
| `tau_m` | 10.0 | Membrane time constant |
| `tau_adapt` | 150.0 | Adaptation time constant |
| `a_adapt` | 0.05 | Subthreshold adaptation coupling |
| `gain` | 2.0 | Baseline input gain |
| `serotonin` | 0.0 | Serotonin modulation level |
| `dt` | 0.5 | Model timestep |

---

## Safety and fidelity contract

The Python model, Rust engine, Go service, Julia kernel, and Rust safety shim
share the same runtime boundary:

- finite state and parameter validation before integration
- positive membrane and adaptation time constants
- positive timestep
- non-negative adaptation coupling and gain
- membrane voltage constrained to the physical `[-100, 60] mV` operating range
- serotonin constrained to `[0, 1]`
- non-negative adaptation current
- threshold above reset and rest potentials
- finite runtime input current
- exact first-order membrane relaxation under constant drive
- exact first-order adaptation relaxation against the non-negative candidate target
- local candidate-state integration with no partial mutation on invalid input

Invalid state or input raises `ValueError` on throwing Python/Julia surfaces and
returns no spike without mutation on non-throwing Go/Rust safety surfaces.

---

## Verification surfaces

Module-specific tests cover bounded state evolution, serotonin firing relation,
invalid-configuration rejection, non-finite-drive preservation, and corrupted
state preservation. Python, Go, Rust safety, Julia, and Rust engine checks also
assert the closed-form membrane/adaptation relaxation contract and fail-closed
nonphysical-voltage boundary.

---

## Benchmark status

The maintained Rust engine path was remeasured on 2026-05-31 after the
closed-form relaxation and fail-closed voltage-boundary hardening:

- command: `cargo bench --manifest-path engine/Cargo.toml --bench full_bench lugaro_10k_steps`
- hardware: Intel Core i5-11600K @ 3.90 GHz, 6C/12T, verified with `lscpu`
- Criterion median: `0.23055588627787307 ms` per 10k steps (`23.05558862778731 ns` per step)
- artifact:
  `benchmarks/results/local_i5_11600k_criterion_2026-05-31_lugaro_cell.json`

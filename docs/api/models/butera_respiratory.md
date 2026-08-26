<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SC-NeuroCore — Butera respiratory neuron model documentation -->

# Butera Model 1 Respiratory Pacemaker

`ButeraRespiratoryNeuron` implements Model 1 from Butera, Rinzel, and Smith
(1999), DOI `10.1152/jn.1999.82.1.382`. Its three dynamic states are membrane
voltage `v`, delayed-rectifier activation `n`, and persistent-sodium
inactivation `h_nap`.

## Source contract

```text
I_Na    = g_Na  m_Na(v)^3 (1-n) (v-E_Na)
I_NaP   = g_NaP m_NaP(v) h_NaP   (v-E_Na)
I_K     = g_K n^4                 (v-E_K)
I_L     = g_L                     (v-E_L)
I_tonic = g_tonic                 (v-E_syn)
C dv/dt = -I_Na-I_NaP-I_K-I_L-I_tonic+I_app
dn/dt   = (n_inf(v)-n)/tau_n(v)
dh/dt   = (h_inf(v)-h)/tau_h(v)
```

The source default is `C=21 pF`; `step(current)` interprets `current` as applied
current `I_app` in pA. Tonic synaptic conductance is separately configurable and
defaults to zero. The continuous paper ODE does not reset on a spike. The API
returns an observational event only for a sampled upward crossing of
`v_threshold`.

The repository integrates the equations with candidate-first RK4 at the
configurable default `dt=0.1 ms`. This is an implementation specialization, not
a claim that the paper prescribed RK4 or that timestep.

## Compatibility identity

The former repository recurrence omitted the whole-cell capacitance divisor.
It remains available as `SCUnitCapacitanceRespiratoryNeuron`, which is
equivalent to the historical `C=1` RK4 behavior. It is count-neutral and carries
no Butera-paper attribution. The SC identity has an independent public
registration, descriptor, paired schema, project receipt, native safety and
runtime surfaces, documentation, and benchmark; it is not a hidden alias.

## Evidence and boundary

- The independent 1,024-step mixed-drive receipt records four events at indices
  168, 447, 692, and 955 and binds every state/event word by SHA-256.
- Python, Rust engine, Rust safety, Go, Julia, and Mojo execute the same source
  contract. The source-bound five-runtime benchmark records 954 exact events
  over 200,000 steps at `I_app=50 pA`.
- The numerically sensitive SC unit-capacitance profile has exact one-step state
  parity to `2e-12`. Its 20,000-step benchmark records five events in
  Python/Rust/Go/Julia and four in Mojo; that explicit one-event `libm` envelope
  is the claimed boundary, not bit-identical long-run state.
- Invalid inputs and non-finite candidates fail before state mutation.
- Benchmark timings are loaded-host regression evidence only. No production
  speed, RTL, synthesis, formal-equivalence, timing, PPA, board/HIL, device, or
  silicon claim is made.

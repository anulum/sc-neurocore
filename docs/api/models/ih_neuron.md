# IhNeuron

`IhNeuron` is an experimental SC-NeuroCore composite: a Wang–Buzsáki
sodium/potassium base with one hyperpolarisation-activated mixed-cation gate.
It is useful for testing voltage sag and rebound behaviour, but it is not a
publication-exact HCN cell model.

## Provenance boundary

- Wang and Buzsáki (1996) supplies the fast-spiking sodium/potassium base:
  <https://doi.org/10.1523/JNEUROSCI.16-20-06402.1996>.
- Robinson and Siegelbaum (2003) reviews HCN molecular and physiological
  function: <https://doi.org/10.1146/annurev.physiol.65.092101.142734>.
- Pape (1996) reviews neuronal hyperpolarisation-activated current:
  <https://doi.org/10.1146/annurev.ph.58.030196.001503>.

The two HCN papers are reviews. Neither defines the exact activation curve,
time constant, conductances, or WB+HCN combination implemented here. The old
“Robinson & Bhatt, Neuron 11:953, 1993” attribution was erroneous and has been
removed.

## Maintained recurrence

The complete state is `(v, h, n, r)`. One public call advances 50 forward-Euler
substeps of 0.01 ms each:

```text
m_inf = alpha_m(v) / (alpha_m(v) + beta_m(v))
dh/dt = phi * (alpha_h(v) * (1 - h) - beta_h(v) * h)
dn/dt = phi * (alpha_n(v) * (1 - n) - beta_n(v) * n)
r_inf = 1 / (1 + exp((v + 80) / 10))
tau_r = 100 + 200 / (1 + exp((v + 70) / 10))
dr/dt = (r_inf - r) / tau_r
C_m dv/dt = -I_Na - I_K - I_h - I_L + gain * I
```

The currents are:

```text
I_Na = g_na * m_inf^3 * h * (v - e_na)
I_K  = g_k  * n^4       * (v - e_k)
I_h  = g_h  * r         * (v - e_h)
I_L  = g_l              * (v - e_l)
```

Crossing `v_threshold` records an event and resets only `v` to -65 mV. Gates
continue evolving. The defaults and public parameter ranges are canonical in
`src/sc_neurocore/neurons/model_descriptors/IhNeuron.toml`.

## Failure semantics

Non-finite drive, invalid configuration, and non-finite candidate state are
rejected before mutation. Python and the production PyO3 binding raise
`ValueError`; the standalone Rust, Go, and Julia surfaces return or throw their
native error. Legacy direct Rust/Go `Step` callers fail closed with no mutation.
Finite accepted trajectories retain the historical recurrence and final
state clamps.

## Executed implementation custody

| Surface | Source | Status |
|---|---|---|
| Python reference | `src/sc_neurocore/neurons/models/ih_neuron.py` | implemented |
| Production Rust | `engine/src/neurons/channels/ih.rs` | implemented and PyO3-exposed |
| Standalone Rust safety | `src/sc_neurocore/accel/rust/safety/ih_neuron.rs` | implemented |
| Go | `src/sc_neurocore/accel/go/services/ih_neuron.go` | implemented |
| Julia | `src/sc_neurocore/accel/julia/neurons/ih_neuron.jl` | implemented |
| Mojo | — | not implemented |
| RTL / synthesis | — | not implemented |

The executed parity contract uses 64 non-constant drives and compares all four
states plus the complete event vector within `1e-12`. The reproducibility
anchor is 1,000 default steps at `I=5`: 135 events, final state
`(-57.67348267126735, 0.5416473882147339, 0.14965651883058248,
0.08783655571240012)`, and big-endian `(v,h,n,r,event)` trace SHA-256
`204dd9c459e4ea16ccf79ad58796376b9f47c20e0e738e416717203c0386d09b`.

No Mojo, RTL, synthesis, timing, device, or PPA result is claimed.

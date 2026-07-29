# PersistentNaNeuron

`PersistentNaNeuron` is an experimental SC-NeuroCore composite: a
Wang–Buzsáki sodium/potassium base with one slowly activating,
non-inactivating persistent-sodium gate. It is not a publication-exact cell
model.

## Provenance boundary

- Wang and Buzsáki (1996) supplies the fast-spiking sodium/potassium base:
  <https://doi.org/10.1523/JNEUROSCI.16-20-06402.1996>.
- French et al. (1990) provides primary physiological evidence for a
  voltage-dependent persistent sodium current in hippocampal neurons:
  <https://doi.org/10.1085/jgp.95.6.1139>.
- Crill (1996) reviews persistent sodium current in mammalian central neurons:
  <https://doi.org/10.1146/annurev.ph.58.030196.002025>.

The former “French et al., Neuroscience 42:363, 1990” citation was incorrect.
Neither the French paper nor the Crill review defines the exact activation
curve, time constant, conductances, or WB+INaP combination implemented here.

## Maintained recurrence

The complete state is `(v, h, n, p)`. One public call advances 50 forward-Euler
substeps of 0.01 ms each:

```text
m_inf = alpha_m(v) / (alpha_m(v) + beta_m(v))
dh/dt = phi * (alpha_h(v) * (1 - h) - beta_h(v) * h)
dn/dt = phi * (alpha_n(v) * (1 - n) - beta_n(v) * n)
p_inf = 1 / (1 + exp(-(v + 48) / 5))
tau_p = 10 + 40 / (1 + ((v + 48) / 10)^2)
dp/dt = (p_inf - p) / tau_p
C_m dv/dt = -I_Na - I_NaP - I_K - I_L + gain * I
```

The currents are:

```text
I_Na  = g_na  * m_inf^3 * h * (v - e_na)
I_NaP = g_nap * p             * (v - e_na)
I_K   = g_k   * n^4           * (v - e_k)
I_L   = g_l                   * (v - e_l)
```

Crossing `v_threshold` records an event and resets only `v` to -65 mV. Gates
continue evolving. Defaults and public parameter ranges are canonical in
`src/sc_neurocore/neurons/model_descriptors/PersistentNaNeuron.toml`.

## Failure semantics

Non-finite drive, invalid configuration, and non-finite candidate state are
rejected before mutation. Python and the production PyO3 binding raise
`ValueError`; standalone Rust, Go, and Julia use their native error surfaces.
Legacy direct Rust/Go `Step` callers fail closed with no mutation. Finite
accepted trajectories retain the historical recurrence and final state clamps.

## Executed implementation custody

| Surface | Source | Status |
|---|---|---|
| Python reference | `src/sc_neurocore/neurons/models/persistent_na_neuron.py` | implemented |
| Production Rust | `engine/src/neurons/channels/persistent_na.rs` | implemented and PyO3-exposed |
| Standalone Rust safety | `src/sc_neurocore/accel/rust/safety/persistent_na_neuron.rs` | implemented |
| Go | `src/sc_neurocore/accel/go/services/persistent_na_neuron.go` | implemented |
| Julia | `src/sc_neurocore/accel/julia/neurons/persistent_na_neuron.jl` | implemented |
| Mojo | — | not implemented |
| RTL / synthesis | — | not implemented |

The executed parity contract uses 64 non-constant drives and compares all four
states plus the complete event vector within `1e-12`. The reproducibility anchor
is 1,000 default steps at `I=5`: 192 events, final state
`(-47.10006085426508, 0.42580153415012734, 0.19451617841811397,
0.2438674108845956)`, and big-endian `(v,h,n,p,event)` trace SHA-256
`006dbe26735718a9db0f6f0c8fa4b18fa7e883aaac4c744776607786b2ee8354`.

No Mojo, RTL, synthesis, timing, device, or PPA result is claimed.

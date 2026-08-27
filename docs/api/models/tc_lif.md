# TwoCompartmentLIFNeuron

**Module:** `sc_neurocore.neurons.models.tc_lif`
**Reference:** Zhang, S., Yang, Q., Ma, C., Wu, J., Li, H. & Tan, K.C. (2024). TC-LIF: A Two-Compartment Spiking Neuron Model for Long-Term Sequential Modelling. *AAAI* 38(15):16838–16847 (doi 10.1609/aaai.v38i15.29625)
**Family:** Compartmental (dendrite + soma integrator map)
**State variables:** `u_d` (dendritic potential), `u_s` (somatic potential), `s_prev` (previous-step spike)

## Equations (paper Eqs. 10–12, exact ordering)

$$U^D[t] = U^D[t-1] + \beta_1 U^S[t-1] + I[t] - \gamma S[t-1]$$
$$U^S[t] = U^S[t-1] + \beta_2 U^D[t] - V_{th} S[t-1]$$
$$S[t] = \Theta\bigl(U^S[t] - V_{th}\bigr)$$

One external current $I[t]$ enters the dendrite. Both compartments
reset **softly** through the delayed spike $S[t-1]$ (the subtraction
terms); there is no hard reset and no leak toward a rest potential.
$\beta_1 \equiv -\sigma(c_1) \in (-1, 0)$ and
$\beta_2 \equiv \sigma(c_2) \in (0, 1)$ are trained per task in the
paper. $\Theta(0) = 1$ (right-continuous) is a repository convention.

## Published profiles (Table 5)

The paper has dataset/network-specific initialisations and **no
universal default**. Defaults here are the S-MNIST feedforward
profile; every Table 5 row is exposed via
`TC_LIF_PROFILES` / `TwoCompartmentLIFNeuron.from_profile(name)`:

| Profile | β₁ | β₂ | γ | V_th |
|---------|----|----|----|------|
| `smnist_feedforward` (default) | −0.5 | 0.5 | 0.5 | 1.0 |
| `smnist_recurrent` | −0.8 | 0.4 | 0.5 | 1.0 |
| `psmnist_feedforward` | −0.5 | 0.5 | 0.7 | 1.5 |
| `psmnist_recurrent` | −0.2 | 0.8 | 0.5 | 1.8 |
| `gsc_feedforward` | −0.5 | 0.5 | 0.6 | 1.2 |
| `gsc_recurrent` | −0.8 | 0.8 | 0.7 | 1.25 |
| `shd_feedforward` | −0.5 | 0.5 | 0.5 | 1.5 |
| `shd_recurrent` | −0.5 | 0.5 | 0.5 | 1.5 |
| `ssc_feedforward` | −0.5 | 0.5 | 0.5 | 1.5 |
| `ssc_recurrent` | −0.5 | 0.5 | 0.5 | 1.5 |

## Invalid-Input Atomicity (Fail-Closed Contract)

`step(i_ext)` validates before touching state: a non-finite input
raises `ValueError` with the pre-step state preserved exactly; an
out-of-bounds configuration (β signs and open intervals enforced)
raises at construction and at each step; runaway accumulation beyond
the public |state| ≤ 1e6 bound is rejected at the start of the next
step. The pure-arithmetic map keeps candidates finite in binary64 for
every valid state and finite input. The production Rust engine
(`try_step`), the typed PyO3 binding, the standalone safety Rust, Go
(`TryStep`), and Julia (`ArgumentError`) enforce the same contract.

## Backend Inventory

| Surface | Status |
|---------|--------|
| Python reference | `src/sc_neurocore/neurons/models/tc_lif.py` |
| Production Rust engine | `engine/src/neurons/multi_compartment/two_compartment_lif.rs` (`try_step`) |
| PyO3 binding | `engine/src/bindings/multi_compartment/two_compartment_lif.rs` (typed `ValueError`, `get_state`) |
| NetworkRunner | `WrTwoCompLIF` adapter: runner drive = the single dendritic input, reported voltage = `u_s` |
| Standalone safety Rust | `src/sc_neurocore/accel/rust/safety/tc_lif.rs` |
| Go service | `src/sc_neurocore/accel/go/services/tc_lif.go` |
| Julia mirror | `src/sc_neurocore/accel/julia/neurons/tc_lif.jl` |
| Mojo | not implemented; no kernel exists and no parity is claimed |
| Silicon / RTL | not implemented; no HDL parity claimed |
| Backend parity | engine, safety Rust, Go, Julia vs Python: 64-step complete state **bit-exact** (atol = 0) |
| Module-owned tests | `tests/test_model_tc_lif_atomicity.py`, `tests/test_tc_lif_backends.py` |

## Preserved historical identities

Two structurally different recurrences were formerly published under
this name and remain available as count-neutral SC identities with
frozen trajectory anchors — see
[SCLeakyTwoCompartmentLIFNeuron](sc_leaky_tc_lif.md) (leaky,
hard-reset, two-current Python recurrence) and
[SCExponentialTwoCompartmentLIFNeuron](sc_exponential_tc_lif.md)
(exponential-decay, hard-reset engine recurrence). Neither is the
Zhang et al. (2024) model; the canonical class above is.

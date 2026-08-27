# NMDA-autapse pyramidal neuron

`NMDANeuron` is a bounded scalar specialization of the pyramidal-cell and
NMDA equations used by Wang (1999). It combines the leaky integrate-and-fire
membrane (Eq. 1), optional calcium-activated potassium term (Eq. 2), and the
two-stage saturating NMDA gate (Eqs. 4–5). The neuron's own emitted events
increment the presynaptic NMDA transmitter state, matching the NMDA-only
autapse configuration of the source single-cell experiment.

Reference: X.-J. Wang, “Synaptic basis of cortical persistent activity: the
importance of NMDA receptors to working memory,” *Journal of Neuroscience*
19(21):9587–9603, 1999, DOI
[`10.1523/JNEUROSCI.19-21-09587.1999`](https://doi.org/10.1523/JNEUROSCI.19-21-09587.1999).
The voltage-dependent magnesium-block factor follows Jahr and Stevens (1990),
DOI
[`10.1523/JNEUROSCI.10-09-03178.1990`](https://doi.org/10.1523/JNEUROSCI.10-09-03178.1990).

## State and equations

The dynamic state is `v`, `x_nmda`, `s_nmda`, `ca`, and
`refractory_remaining`. With applied current `I_app`,

\[
C_m\dot V=-g_L(V-V_L)-g_{NMDA}sB(V)(V-E_{NMDA})
-g_{AHP}Ca(V-V_K)+I_{app},
\]

\[
\dot x=-x/\tau_x,\qquad
\dot s=\alpha_s x(1-s)-s/\tau_s,\qquad
\dot {Ca}=-Ca/\tau_{Ca},
\]

\[
B(V)=\frac{1}{1+[Mg]e^{-0.062V}/3.57}.
\]

An emitted event increments `x_nmda` by `alpha_x` and `ca` by
`alpha_ca`. Source defaults are `C_m=0.5 nF`, `g_L=0.025 uS`,
`V_L=-70 mV`, `V_threshold=-52 mV`, `V_reset=-59 mV`, a `2 ms`
refractory period, `tau_x=2 ms`, `tau_s=80 ms`, and `g_nmda=0.1 uS`.
The optional AHP term is inactive by default (`g_ahp=0`).

## Numerical boundary

The source reports second-order Runge–Kutta at `0.02–0.05 ms` with
interpolated spike times. This implementation uses midpoint RK2 at the source
upper grid bound, `dt=0.05 ms`, and sampled upward threshold detection. That
sampled-grid event rule is an explicit implementation specialization; it is not
presented as spike-time interpolation.

The identity is intentionally scalar and NMDA-only. It does not claim the
paper's full excitatory/inhibitory network, AMPA/GABA populations, parameter
sweeps, or exact reproduction of Figure 3 firing rates.

## Maintained surfaces and evidence

- Python: `src/sc_neurocore/neurons/models/nmda_neuron.py`
- production Rust and PyO3: `engine/src/neurons/channels/nmda.rs` and
  `engine/src/bindings/channels/nmda_neuron.rs`
- standalone safety Rust, Go, Julia, and executable Mojo mirrors under
  `src/sc_neurocore/accel/`
- paired TOML/JSON schema, curated descriptor, independent 512-step receipt,
  complete-state native parity, and a source-hashed five-runtime benchmark
- source-default signed-Q16.16 midpoint-RK2 RTL, an independent bit-exact
  integer oracle, Yosys synthesis, and a depth-4 CVC5 bounded-safety job

Invalid state, configuration, or non-finite input is rejected before state is
committed. The production engine, NetworkRunner factory, and PyO3 class expose
the source identity directly.

The enrolled RTL replaces the exponential magnesium factor with linear
interpolation between 5 mV Q16.16 LUT samples over `[-120, 80] mV`. On the
512-step mixed-current source receipt it preserves all four events at their
exact indices; maximum absolute errors are below `0.012 mV` for voltage,
`0.0004` for `x_nmda`, `0.0006` for `s_nmda`, `0.0024` for calcium, and
`0.00013 ms` for refractory time. It is bit-exact to its independent integer
oracle and synthesizes in Yosys. The formal job proves bounded reset and public
state safety at depth 4; it does not prove binary64 equation equivalence.

Configurable-parameter RTL, timing, PPA, device, and silicon validation remain
outside the claim.

## Preserved project recurrence

The former WB membrane plus input-driven NMDA implementation remains available
without publication misattribution as
[`SCWBNMDAMagnesiumBlockNeuron`](sc_wb_nmda_magnesium_block.md). It is a
count-neutral compatibility identity, not an alias for this source model, and
carries its own separately bounded Q32.32 FSM evidence.

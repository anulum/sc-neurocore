# SCScaledResetAdaptiveIFNeuron

**Module:** `sc_neurocore.neurons.models.sc_scaled_reset_adaptive_if`

**Identity:** count-neutral SC-NeuroCore retained project recurrence; no whole-model publication attribution

**State:** `v`, `theta`, `i1`, `i2`

## Equations and dynamics

This class preserves the former Model 14 behaviour under an explicit SC identity:

$$
\dot v=\frac{-(v-v_{rest})+i_1+i_2+I}{\tau_v},\quad
\dot\theta=\frac{\theta_\infty-\theta+a(v-v_{rest})}{\tau_\theta},\quad
\dot i_j=-\frac{i_j}{\tau_j}.
$$

A finite RK4 candidate with `v >= theta` installs
`v = v_reset + b*(v-v_rest)`, `theta = max(theta, theta_reset)`, `i1 += r1`, and `i2 += r2`.

## Parameters and defaults

The public compatibility defaults are `v=v_rest=v_reset=0`,
`theta=theta_reset=theta_inf=1`, `tau_v=tau_1=10 ms`, `tau_theta=100 ms`,
`tau_2=200 ms`, `a=b=r1=r2=0`, and `dt=1 ms`. The enrolled receipt profile
sets `theta_reset=1.3`, `tau_theta=40`, `tau_1=15`, `tau_2=80`, `a=b=0.1`,
`r1=0.2`, and `r2=-0.15`.

## Verification and benchmark

Python, production Rust/PyO3, Rust safety, Julia, Go, and Mojo preserve the recurrence and failure-atomic validation. The independent 1,600-step mixed-drive receipt records 168 events and SHA-256 `6d2aadb7…e78cf`; the focused 200,000-step benchmark is exact across all five runtime traces and events.

Committed Q16.16 RTL preserves the complete 250-step `I=3` event vector (31 events) and keeps all four states within `0.001`. Icarus compilation, Yosys coarse synthesis, and the source-profile depth-2 SymbiYosys/Z3 reset-spike safety job pass. Timing, PPA, device, board, physical silicon, and universal real-number equivalence remain open.

<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
# BertramPhantomBurster

`BertramPhantomBurster` implements the four-state pancreatic beta-cell phantom
burster of Bertram, Previte, Sherman, Kinard, and Satin (2000), equations 1–10.
The parameter set and initial state follow the authors' `BJ_00.ode` program.

Reference: [doi:10.1016/S0006-3495(00)76525-8](https://doi.org/10.1016/S0006-3495(00)76525-8).

## State and currents

The state is `(V, n, s1, s2)`. Unlike the former project implementation, the
fast potassium gate `n` is dynamic:

$$\dot V = (-I_{Ca}-I_K-I_{s1}-I_{s2}-I_L+I_{ext})/C_m$$

$$\dot n = \lambda_n(n_\infty(V)-n)/\tau_n(V)$$

$$\dot s_1=(s_{1,\infty}(V)-s_1)/\tau_{s1},\qquad
\dot s_2=(s_{2,\infty}(V)-s_2)/\tau_{s2}$$

with

$$I_{Ca}=g_{Ca}m_\infty(V)(V-V_{Ca}),\quad I_K=g_Kn(V-V_K),$$

$$I_{s1}=g_{s1}s_1(V-V_K),\quad I_{s2}=g_{s2}s_2(V-V_K),\quad
I_L=g_L(V-V_L).$$

Each steady-state gate is `1/(1+exp((midpoint-V)/slope))`, and
`tau_n=tau_n_bar/(1+exp((V-v_n)/s_n))`.

## Author-code defaults

| Field | Default |
|---|---:|
| `(v, n, s1, s2)` | `(-43, 0.03, 0.1, 0.434)` |
| `(g_ca, g_k, g_s1, g_s2, g_l)` pS | `(280, 1300, 20, 32, 25)` |
| `(e_ca, e_k, e_l)` mV | `(100, -80, -40)` |
| `c_m` fF | `4524` |
| `(v_m, s_m)` mV | `(-22, 7.5)` |
| `(v_n, s_n)` mV | `(-9, 10)` |
| `(v_s1, s_s1)` mV | `(-40, 0.5)` |
| `(v_s2, s_s2)` mV | `(-42, 0.4)` |
| `(tau_n_bar, tau_s1, tau_s2)` ms | `(9.09, 1000, 120000)` |

The production specialization uses simultaneous fixed-step RK4 at `dt=0.5 ms`.
The authors selected adaptive CVODE, so the implementation claims equation and
parameter fidelity, not identical adaptive-solver interpolation. External
`current` is an additive extension. Events are sampled upward crossings of
`v_threshold=-20 mV` and do not reset the continuous state.

## Identity boundary

The previous three-state recurrence used instantaneous `n_inf`, different
conductances and reversals, and different slow time constants. It remains
available, count-neutrally and without paper attribution, as
`SCThreeStatePhantomBurster`.

## Evidence

- paired TOML/JSON schemas record the complete state, parameters, integrator,
  sampled event rule, and identity boundary;
- the independent receipt pins a 512-step mixed-drive trace, three events, all
  four final states, and SHA-256;
- Python is checked against an independently written RK4 oracle;
- Rust, Julia, Go, and Mojo execute the same four-state equations and match the
  enrolled one-step state within the declared `5e-13` transcendental tolerance.

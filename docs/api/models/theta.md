# ThetaNeuron

**Module:** `sc_neurocore.neurons.models.theta`
**Reference:** Ermentrout and Kopell (1986), DOI `10.1137/0146017`
**Family:** canonical Type-I phase model on the unit circle
**Source scope:** equation (2.5), or equation (3.3) with a frozen slow drive
**State:** `theta`, normalised to `[-pi, pi)`

## Dynamics

The maintained hand model follows

$$
\frac{d\theta}{dt} = (1 - \cos\theta) + (1 + \cos\theta)I.
$$

This is Ermentrout and Kopell's equation (2.5), with `I` representing their
dimensionless parameter `a`. It also represents equation (3.3) when the slow
drive `g(0,y,0)` is held constant for one update. The class does **not** claim
the full coupled slow oscillator and parabolic-bursting construction in
equations (3.3)-(3.7). The inspected author-hosted paper scan and the complete
source protocol are SHA-256-bound in
`reference_receipts/theta_ermentrout_kopell_1986.json`.

With the tangent half-angle substitution $y=\tan(\theta/2)$, the equation becomes

$$
\frac{dy}{dt}=y^2+I.
$$

`ThetaNeuron.step()` integrates this transformed equation analytically for a
constant current during each timestep. It does not use a forward-Euler phase
increment.

- For $I>0$, the flow advances
  $\operatorname{atan}(y/\sqrt I)$ by $\sqrt I\,dt$.
- For $I=0$, the flow is $y' = y/(1-y\,dt)$.
- For $I<0$, with $a=\sqrt{-I}$ and $r=(y-a)/(y+a)$, the flow advances
  $r' = r\exp(2a\,dt)$ and recovers $y'=a(1+r')/(1-r')$.

The result is mapped back with $\theta'=2\operatorname{atan}(y')$ and wrapped
onto the compact circle. A spike is reported when the analytic trajectory
crosses $\pi$ within the timestep. The state remains wrapped after that event;
there is no separate voltage-style reset parameter.

Because the public event packet is binary per sample, a positive-drive step
must satisfy $\sqrt I\,dt\leq\pi$. A larger held step can contain more than one
source passage. Every runtime rejects that unrepresentable request before
mutating state instead of silently compressing multiple spikes into one bit.

For positive current, the continuous-time period is

$$
T=\frac{\pi}{\sqrt I},\qquad f=\frac{\sqrt I}{\pi}.
$$

For negative current the stable phase is

$$
\theta^*=-\arccos\!\left(\frac{1+I}{1-I}\right).
$$

## Public parameters and state

| Field | Default | Contract |
|---|---:|---|
| `theta` | `0.0` | finite phase; normalised at construction |
| `dt` | `0.01` | finite and strictly positive |
| `current` | `0.0` | finite dimensionless source parameter `a`, held for each call |

Every runtime path rejects invalid current, phase, timestep, or non-finite
analytic candidates before committing state. `reset()` restores `theta=0.0`
while preserving `dt`.

## Acceleration and dispatch

`simulate_complete()` returns aligned post-step phase and `uint8` event arrays
and commits the checked final phase only after the whole packet validates.
`simulate()` is the aggregate-count compatibility view over that same packet.
Python, production Rust/PyO3, safety Rust, Julia, Go, and Mojo transport the
complete initial-phase and timestep contract. The Go and Mojo C ABIs stage both
output buffers and leave them untouched on rejection.

Auto dispatch probes Go, Julia, Mojo, compatible Rust, then Python. Go is
probed before Julia so a built shared library avoids initialising the Julia
runtime. The controlled benchmark records both this stable dispatcher order
and the timing order observed during the individual host run.

The enrolled 1,000-step current vector is:

| Current | Events |
|---:|---:|
| `-1.0` | 0 |
| `-0.5` | 0 |
| `0.0` | 0 |
| `0.1` | 1 |
| `0.333` | 2 |
| `0.5` | 2 |
| `1.0` | 3 |
| `2.0` | 5 |
| `5.0` | 7 |
| `20.0` | 14 |
| `50.0` | 23 |

All four compiled lanes preserve these event counts exactly. Across that
vector, the measured circular phase difference from Python is below `2e-12`.
On the current host, Rust and Mojo were bit-identical to Python; Julia and Go
reached maximum circular differences of approximately `2.1e-13`. The declared
portable contract remains the `2e-12` envelope because transcendental library
implementations can differ by a small number of ULPs.

## Schema and generated RTL boundary

The Python hand model and acceleration chain use the analytic constant-current
flow. The paired `theta.toml` and `theta.json` schema/compiler surfaces retain
an explicit Euler step, a phase threshold at `3.11`, and subtraction of
`6.2832` on a crossing. That is an intentional fixed-point hardware
approximation, not an exact-flow or bit-true claim.

The generated Q16.16 RTL preserves the complete event-count vector above over
1,000 steps. For currents `-1`, `-0.5`, `0`, `0.333`, `0.5`, `1`, and `2`, its
maximum shortest-arc phase error remains below `0.17` rad. At `I=1`, all three
events are preserved but each is displaced by one cycle, producing six binary
event-sample differences. The higher-current points retain event-count evidence
only; no bounded trajectory claim is made for those rapidly rotating traces.

The committed `sc_theta` Q16.16 design synthesizes with Yosys 0.33 to 6,203
coarse cells, including 33 negative-reset DFF bits and 381 multiplexers. Its
source-labelled depth-110 Z3 job reaches the first receipt-drive event and
proves reset, the declared bounded phase envelope, and negative-side wrap on
an event under the enrolled `I=2` fixed-drive formal contract.
This is H2 evidence, not timing, PPA, board, physical-silicon, or universal
equivalence evidence.

## Controlled local benchmark

The committed artefact is
`benchmarks/results/local_python_2026-06-16_theta_exact_flow.json`, generated by
`benchmarks/bench_model_theta.py`. It measures 100,000 public-dispatch steps at
`I=0.5`, seven repeats, with affinity pinned to one logical CPU. The workstation
was not otherwise isolated and was under concurrent load, so these numbers are
local regression evidence rather than production throughput claims.

| Backend | Median call (ms) | Minimum call (ms) | Events | Max circular difference |
|---|---:|---:|---:|---:|
| Rust/PyO3 | 10.462 | 9.890 | 225 | `0` |
| Julia | 11.396 | 10.958 | 225 | `7.20e-14` |
| Go | 13.946 | 11.989 | 225 | `1.99e-13` |
| Mojo | 24.357 | 24.012 | 225 | `0` |
| Python | 126.126 | 117.079 | 225 | `0` |

The artefact records runtime versions, CPU affinity, governor, host load,
source hashes for the runtime, ABI, factory, receipt, descriptor, schema, RTL,
formal, synthesis, and readiness surfaces; final phase; the identical complete
event-vector digest `9dc6d01a...ebe4fe4d`; and a successful Rust-safety test
receipt. The pinned run used logical CPU 10 and reported host load rather than
claiming an isolated benchmarking host.

## Verification surfaces

- `tests/test_model_theta_theta_*.py`: analytic dynamics, phase geometry,
  validation, network integration, complete packets, and public simulation.
- `tests/test_theta_backends_*.py`: executable four-lane complete-packet
  parity, full-parameter ABIs, fail-closed buffers, fallback order, and the
  Rust-safety trace probe.
- `tests/test_theta_backend_loading.py`: optional-runtime loading failures.
- `tests/test_reference_theta.py`: PDF-bound receipt digests and independent
  analytic circle-flow/event oracle.
- `tests/test_cosim_theta.py`: paired schemas, Q16.16 event counts, circular
  phase bounds, declared timing boundary, and Yosys receipt.
- `tests/test_bench_theta.py`: controlled benchmark and source-hash contract.
- `src/sc_neurocore/accel/julia/theta_parity_test.jl`: standalone Julia event
  vector, configured-state, empty-run, and rejection assertions.
- `src/sc_neurocore/accel/rust/safety/theta.rs`: standalone exact-flow safety
  unit tests.

The touched Python dispatcher and backend wrapper each have focused 100%
statement coverage.

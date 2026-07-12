# MedvedevMapNeuron

`MedvedevMapNeuron` is the scalar slow-calcium first-return reduction derived
from Medvedev (2005), not a tent map or a modulo-one chaotic circle map. The
maintained model follows the three asymptotic regions constructed in Section 4
of the paper and exposes every calibration constant explicitly.

**Python:** `sc_neurocore.neurons.models.medvedev_map.MedvedevMapNeuron`

**Rust engine:** `engine/src/neurons/maps.rs::MedvedevMapNeuron`

**Reference:** G. S. Medvedev, *Reduction of a model of an excitable cell to a
one-dimensional map*, Physica D 202 (2005), 37–59,
[DOI 10.1016/j.physd.2005.01.021](https://doi.org/10.1016/j.physd.2005.01.021)

## Source and calibration boundary

The paper derives a return map for the slow calcium variable (u), but it does
not provide one uniquely tabulated pair of global functions (T(u)) and
(F(u)). SC-NeuroCore therefore distinguishes two layers:

- the branch construction and bifurcation boundaries come from Eqs. 4.4, 4.7,
  4.8, 4.13, and 4.15;
- `decay_t0`, `alpha_t0`, `f_0`, `f_1`, `homoclinic_exponent`, and `d` are the
  disclosed reproducible calibration of that construction.

The cited source model is recovered with `current=0`. Non-zero `current` is an
SC-NeuroCore perturbation applied only to active returns; it is not attributed
to the paper.

## Recurrence

The three source boundaries are derived from the bifurcation parameters:

\[
u_0=\frac{\beta_0}{\delta-\beta_0},\qquad
u_{HC}=\frac{\beta_{HC}}{\delta-\beta_{HC}},\qquad
u_{SN}=\frac{\beta_{SN}}{\delta-\beta_{SN}}.
\]

For the inner branch define

\[
u_1=(1-\alpha_{T0})u+\alpha_{T0}f_0,
\qquad
g=\beta_{HC}-\frac{\delta u_1}{1+u_1},
\]

and

\[
R_{HC}(u)=
\begin{cases}
f_1, & g\le 0,\\
\exp\!\left(\eta\log(dg)\right)(u_1-f_1)+f_1, & g>0,
\end{cases}
\]

where (eta) is `homoclinic_exponent`. One map iteration is

\[
u_{n+1}=
\begin{cases}
q_{T0}u_n+(1-q_{T0})f_0+g_I I_n, & u_n\le u_0,\\
R_{HC}(u_n)+g_I I_n, & u_0<u_n\le u_{HC},\\
u_{SN}, & u_n>u_{HC}.
\end{cases}
\]

The left branch is the calibrated exponential-relaxation form of Eq. 4.4
(Eq. 4.7 is its leading small-parameter form), the inner branch composes Eqs.
4.8 and 4.13, and the right branch is the exact Eq. 4.15 return.

## Event convention

An event denotes an active fast-cycle return and is evaluated from the
**pre-step** state:

\[
e_n = \mathbf{1}[u_n\le u_{HC}].
\]

This is a maintained observation convention, not a voltage threshold or a
source-paper spike definition. The right branch emits no event, returns to
(u_{SN}), and does not apply external current.

## Defaults

| Field | Default | Meaning |
|---|---:|---|
| `u` | `0.2514078836724436` | Initial slow state, equal to (u_{SN}) |
| `beta_0` | `0.0015` | Defines (u_0) |
| `beta_hc` | `0.00203` | Defines the homoclinic boundary (u_{HC}) |
| `beta_sn` | `0.002009000318382601` | Defines the saddle-node return (u_{SN}) |
| `delta` | `0.01` | Slow calcium-removal coefficient |
| `decay_t0` | `0.9903563355786734` | Calibrated Eq. 4.4 relaxation factor |
| `alpha_t0` | `0.0096904656865853` | Calibrated Eq. 4.8 affine-return coefficient |
| `f_0` | `1.4713541429802286` | Active-branch fast-subsystem average |
| `f_1` | `0.1820152787145665` | Homoclinic-boundary fast-subsystem average |
| `homoclinic_exponent` | `0.02149298991339221` | Eq. 4.13 exponent |
| `d` | `2271.1927977404063` | Eq. 4.13 log-argument scale |
| `input_gain` | `0.01` | Maintained active-return perturbation gain |

The constructor and every simulation call validate finiteness and the topology
(0<\beta_0<\beta_{SN}<\beta_{HC}<\delta), coefficient ranges, calibration
ordering, and positive homoclinic parameters. A rejected step leaves `u`
unchanged.

## Python API

```python
from sc_neurocore.neurons.models.medvedev_map import MedvedevMapNeuron

neuron = MedvedevMapNeuron()
trace, events = neuron.simulate(100, current=2.0, backend="auto")

assert events == 75
assert neuron.u == neuron.beta_sn / (neuron.delta - neuron.beta_sn)
```

`step(current)` advances one return and yields `0` or `1`. `simulate()` returns
the post-step `u` trace plus the event count. `reset()` recomputes (u_{SN})
from the current parameters while preserving the calibration.

## Reproducibility anchors

- With zero current, 100 iterations produce 100 events, final
  `u=0.19448491761002404`, and mean `u=0.21623098362239998`.
- With `current=2`, 100 iterations produce the exact 75-event four-state cycle
  `[0.20201527871456648, 0.23396543697847846, 0.26318342915295445,
  0.2514078836724436]`.
- The little-endian float64 SHA-256 for the 1000-step `current=2` trace is
  `4e45193f652b8c4ab1fc860b179585a52c565cfbe1769b17e850ab770a232f2c`.

The committed DOI feature contract is
`medvedev_map_first_return_doi`; it independently re-derives the equations
without calling the hand model.

## Compiled and silicon paths

Python, the Rust engine, Julia, Go, and Mojo implement the same checked ABI.
Rust, Julia, and Go are bit-identical to Python on the recorded host. Mojo
preserves exact event counts with a measured maximum absolute trace difference
of `4.08e-13` over the parity envelope.

The schema compiler uses Q16.16 for this model because `d=2271.19` cannot fit a
signed Q8.8 word. The shared positive-domain log LUT has 256 points over
`[1/256, 8 + 1/256)` at step `1/32`. At `current=2` over 100 iterations,
generated Q16.16 RTL preserves the complete 75-event vector with maximum state
error below `0.007813`. The corresponding depth-4 Z3 BMC proves reset-spike
safety for the generated `sc_medvedev_map` job.

The pinned 500,000-iteration benchmark is recorded in
`benchmarks/results/bench_medvedev_map.json`; it includes affinity, load,
runtime versions, parity, event counts, and exact source hashes for every
committed kernel surface.

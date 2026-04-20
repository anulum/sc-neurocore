# Julia ODE Solvers

Continuous-time reference solvers implemented in Julia via
`DifferentialEquations.jl`. The suite acts as a numerical ground truth
for the stochastic computing / fixed-point neuron implementations in
`sc_neurocore.neurons` — any SC-path divergence against the Julia ODE
trace is a regression.

```python
from sc_neurocore.accel.julia.solvers import JuliaFusionSolver

solver = JuliaFusionSolver()             # raises if `julia` not on PATH
solver.run_dynamics(steps=2000)          # shells out to fusion_solver.jl
```

---

## 1. `JuliaFusionSolver`

Thin Python façade that locates the Julia binary on `PATH` and invokes
`fusion_solver.jl` with the requested number of integration steps.

| Method                    | Purpose                                                                 |
| ------------------------- | ----------------------------------------------------------------------- |
| `__init__()`              | Resolves `julia` via `shutil.which`; raises `FileNotFoundError` if missing. |
| `run_dynamics(steps: int)`| Runs `julia fusion_solver.jl <steps>`; re-raises on non-zero exit.      |

No in-process Julia binding — the subprocess model keeps the dependency
optional and avoids PyCall's global-state pitfalls.

---

## 2. Solver bundle (`accel/julia/solvers/*.jl`)

### 2.1 `fusion_solver.jl` — reference neuron ODEs

Built on `DifferentialEquations.jl` + `LinearAlgebra`. Canonical
solvers included:

- **LIF** — Leaky Integrate-and-Fire, `τ · dV/dt = -(V − V_rest) + I`.
- **Izhikevich** — 2D quadratic + recovery with the four preset
  regimes `RS`, `IB`, `CH`, `LTS` (parameters per Izhikevich 2003).
- Additional ODEs for HH-style, FitzHugh-Nagumo and Morris-Lecar
  variants — see the file header for the live inventory; the Python
  façade delegates blind, so any solver reachable from the script is
  exercisable.

All solvers use `Tsit5()` (5th-order Runge-Kutta) by default, with
dense output at `saveat=0.1` so the returned trajectory is comparable
to the SC path tick-by-tick.

### 2.2 `neuron_zoo.jl` — population-scale catalogue

Parametric generators for each neuron family above, batched over
`n_neurons` in a single ODE solve — used to cross-check the population
dynamics (mean firing rate, ISI distribution) against the SC / Rust
path.

### 2.3 `dynamical_analysis.jl` — phase-plane diagnostics

Fixed-point finders, Jacobian analyses and nullcline extractors for the
2D neurons (Izhikevich, FitzHugh-Nagumo, Morris-Lecar). Produces the
bifurcation diagrams that doc pages cite.

### 2.4 `spike_analysis.jl` — spike-train metrics

Post-hoc analysis of spike trains emitted by the solvers: inter-spike
interval histograms, Fano factor, population CV, synchrony indices
(van-Rossum, SPIKE-distance). Matches the Python
`sc_neurocore.analysis.spike_stats` surface so any divergence between
the two is a numerical regression signal, not a semantic one.

---

## 3. Toolchain expectations

- Julia 1.9+ on `PATH`.
- `DifferentialEquations.jl` and `LinearAlgebra` in the project's
  Julia env. Install via `julia -e 'using Pkg;
  Pkg.add("DifferentialEquations")'` on first use.
- No Julia state is kept between invocations — each `run_dynamics`
  call is a fresh Julia process. Latency is dominated by JIT warm-up
  (~5-10 s on cold cache); repeated runs within a session amortise.

---

## 4. Limitations

- Subprocess model — not suitable for per-tick calls. Use for offline
  reference traces, batch parameter sweeps, or CI-gated regression
  checks.
- No automatic parameter passthrough — parameters live inside the `.jl`
  scripts. Edit the script or add a CLI layer in `fusion_solver.jl`
  if you need Python-side parameterisation beyond `steps`.
- Output parsing is the script's responsibility; the Python façade
  does not capture stdout by default. Redirect in the shell if you
  need the trajectory back in Python.

---

## Reference

- Python façade: `src/sc_neurocore/accel/julia/solvers/__init__.py`.
- Julia sources: `src/sc_neurocore/accel/julia/solvers/{fusion_solver,neuron_zoo,dynamical_analysis,spike_analysis}.jl`.

::: sc_neurocore.accel.julia.solvers
    options:
      show_root_heading: true

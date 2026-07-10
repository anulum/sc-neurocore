<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
# Conductance-Oscillator Co-Simulation Fidelity

This page records the measured Python-versus-Verilog co-simulation fidelity of
the conductance-based edge-crossing oscillators enrolled in the all-models
co-simulation harness, together with the architectural findings that emerged
while enrolling them. It exists so the fidelity claims are documented per model
and per layer rather than folded into a single headline number: the layers do
not achieve uniform bit-exactness, and the honest statement is that each model
reaches the fidelity its conditioning and the hardware transcendental resolution
allow.

The three co-simulation layers are:

1. **hand** — the maintained Python reference class
   (`HodgkinHuxleyNeuron`, `MorrisLecarNeuron`, `ConnorStevensNeuron`,
   `WangBuzsakiNeuron`).
2. **schema** — the schema-driven `UniversalNeuron` runner
   (`EquationNeuron`) built from `neurons/model_schemas/<model>.toml`.
3. **Q16.16 RTL** — the SystemVerilog datapath emitted by
   `compiler.verilog_compiler`, executed under Icarus Verilog (`iverilog
   -g2012` + `vvp`) with a 32-bit word and 16 fraction bits.

## Per-model, per-layer fidelity

Fidelity is stated at the **spike-count** level (number of rising-edge threshold
crossings over a fixed driven protocol), which is the harness contract for these
non-resetting oscillators. State-level bit-exactness is a stronger property that
only the polynomial and piecewise-linear oscillators achieve; the conductance
models diverge at the least-significant bits because the hand class evaluates
transcendentals through Python `math` while the schema runner uses `numpy`, and
the two travel through distinct Runge-Kutta drivers.

| Model | States | Integrator | hand == schema | schema == Q16.16 RTL |
|-------|--------|------------|----------------|----------------------|
| `morris_lecar` | 2 | single RK4 step per call | exact spike count (7 @ `I=100`, 3000 steps) | exact over the whole `I in [90, 110]` band |
| `hodgkin_huxley` | 4 | 100-sub-step macro RK4 (`substeps=100`) | exact spike count (5 @ `I=20`, 60 macro steps) | exact over a bounded window (`I=15`, `macro=20`); `±1` beyond it |
| `connor_stevens` | 6 | 100-sub-step macro RK4 (`substeps=100`) | exact spike count (10 @ `I=100`, 60 macro steps) | exact over a bounded window (`I=100`, `macro=20`); `±1` beyond it |

Morris-Lecar is well conditioned: it takes a single RK4 step per `step()` call,
its `tanh`/`cosh` gating maps onto the fixed-point look-up datapath without
losing a crossing, and the three layers agree on the spike count exactly across
the whole enrolled current band. Connor-Stevens is a stiff six-state model whose
schema runner reproduces the hand model's spike count exactly once the macro-step
mode is used, but whose Q16.16 datapath holds the exact count only over a bounded
window and drifts by a single spike beyond it. The drift is root-caused below and
is **not** closed by widening the datapath.

## Finding 1 — the conductance hand models macro-step

The maintained conductance reference classes advance a fixed number of inner
integration sub-steps inside one public `step()` call and take **one** threshold
decision at the macro boundary:

- `HodgkinHuxleyNeuron` and `ConnorStevensNeuron`: 100 inner `dt = 0.01`
  sub-steps per 1 ms macro step (`round(1.0 / dt)`).
- `WangBuzsakiNeuron`: 50 inner `dt = 0.01` sub-steps per 0.5 ms macro step
  (`int(0.5 / dt)`).
- `MorrisLecarNeuron` is the exception: a single integration step per call.

A schema runner that took a single `dt` step per `step()` could not reproduce
the sub-stepping models' action-potential count — it over-counted roughly one
crossing per sub-step because every inner step was exposed to the threshold
comparator. This motivated the macro-step integration mode.

## Finding 2 — macro-step integration mode (`[integration] substeps`)

The schema DSL now accepts `substeps = N` in the `[integration]` table. The
runner (`EquationNeuron`) advances `N` inner integration steps via
`_integrate_once` and then takes a single macro-boundary spike decision, matching
the hand models' structure. The Verilog emitter lowers the same semantics with a
sub-step counter (`_ss_cnt`), performing one integration step per clock and
gating both the spike output and the rising-edge history register to the macro
boundary (`_macro_boundary`).

`substeps = 1` (the default) is byte-identical to the previous single-step
datapath, so every already-enrolled model is unchanged. The emitter lowering is
proven bit-exact against the runner on the polynomial FitzHugh-Nagumo oscillator
at `substeps = 1`, `2`, and `4` (runner and Verilog both count 8 crossings) — a
model with no transcendental look-up table to confound the comparison. The
emitter path is guarded to the rising-edge / no-reset / non-pipelined case and
raises `NotImplementedError` otherwise.

## Finding 3 — the Connor-Stevens residual is LUT-resolution-limited, not datapath-precision-limited

Widening the fixed-point word does **not** close the Connor-Stevens spike-count
band: the emitted datapath produces the identical spike count at Q16.16, Q24.24,
and Q32.32. The limiting factor is the resolution of the hardware transcendental
look-up tables, not the number of fraction bits.

The emitter (`compiler.verilog_expr_emitter`) lowers `exp`, `exprel`, `tanh`,
and `cosh` through 256-entry look-up tables sampled over `[-16, 16)` with a step
of `0.125` (`lut_min = -16.0`, `lut_step = 0.125`); `sqrt`, `log`, and the
cube-root `a`-gate use their own ranges. For the stiff six-state Connor-Stevens
gating, that `0.125`-resolution sampling — not the datapath precision — sets the
point at which a marginal crossing is gained or lost. Closing the band would
require finer transcendental tables, which is a separate emitter axis, not a
precision knob. This is documented so a future reader does not spend effort
adding fraction bits that cannot help.

## Finding 4 — integration order matters, not only the macro-step

Reproducing a hand model's spike count requires matching its integration
**order**, not merely its sub-step count:

- **Simultaneous** (Jacobi-style): every derivative is evaluated from the single
  pre-step state, then all states advance together. The schema runner's `euler`
  and `rk4` modes are simultaneous. Connor-Stevens' hand class uses simultaneous
  RK4, so the schema `rk4 + substeps=100` path matches it.
- **Gauss-Seidel** (sequential): the gates are updated first from the old
  membrane voltage, then the membrane current is computed from the **new** gate
  values and the old voltage. The Hodgkin-Huxley default integrator
  (`baseline_euler`) and the Wang-Buzsaki integrator are Gauss-Seidel, so the
  schema's simultaneous integration does **not** reproduce them.

Consequences for the conductance enrolments:

- `hodgkin_huxley` additionally exposes an `integrator="rk4"` option (a
  simultaneous `RK4Solver`), which the schema `rk4 + substeps=100` path matches
  exactly — this is the enrolled operating point (`hand == schema` at `I=20`, 60
  macro steps). The default `baseline_euler` path is Gauss-Seidel and would need a
  sequential schema mode to reproduce; enrolling against the higher-fidelity `rk4`
  path is faithful (same equations, a better integrator) and avoids that.
- `wang_buzsaki` has no RK4 option, so it needs a sequential (Gauss-Seidel)
  schema integration mode before it can be enrolled faithfully — still outstanding.

Both are stiff, so a bounded-window `±1` Q16.16 band comparable to Connor-Stevens
is expected: Hodgkin-Huxley confirms it (exact three-way at `I=15`, `macro=20`;
`±1` beyond), and Wang-Buzsaki is expected to match once its sequential mode lands.

## Provenance corrections landed in-line

While enrolling Connor-Stevens two pre-existing defects were found and fixed in
the same change rather than deferred:

- **Wrong DOI.** The `connor_stevens` schema cited
  `10.1113/jphysiol.1971.sp009368`, which resolves at Crossref to a 1971
  endotoxin-shock paper. The correct reference is
  `10.1113/jphysiol.1971.sp009366` (Connor & Stevens 1971, *Prediction of
  repetitive firing behaviour from voltage clamp data on an isolated neurone
  soma*), Crossref-verified before landing.
- **TOML/JSON rate-form drift.** `connor_stevens.json` carried the singular
  `a*(V - V0) / (1 - exp(...))` rate form (undefined `0/0` at `V = V0`) while the
  loaded `.toml` used the stable `exprel` rewrite. The loader prefers `.toml`, so
  the JSON was dead but inconsistent; it was synchronised to the `exprel` form.

Both corrections reinforce a standing rule: verify every DOI at Crossref before
it lands, and keep the paired TOML/JSON schema forms in sync.

## Known debt (out of scope for the conductance enrolments)

`src/scpn_neurocore/` — a separate three-file SCPN bridge package
(`datastream.py`, `bridge.py`) that is **not** in the CI strict-mypy gate
(`mypy --strict src/sc_neurocore/`) — carries 31 pre-existing `type-arg`
strict-mypy errors. It is recorded here so it is not lost, but it is its own unit
of work and must not be folded into a `sc_neurocore` change.

## Verification

The conductance co-simulation parity is exercised by:

```bash
PYTHONPATH=src python -m pytest tests/test_cosimulation.py -q -k "morris_lecar or connor_stevens or macrostep"
```

The macro-step emitter bit-exactness (FitzHugh-Nagumo, no look-up table) is
exercised by the `TestMacroStepSubstepEmitter` class in the same module. The
schema reference traces for both oscillators are validated by
`tests/test_reference_traces.py` (see
[Reference Trace Harness](reference_traces.md)).
